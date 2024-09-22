import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class RankingDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query = self.data.iloc[idx]['query']
        document = self.data.iloc[idx]['document']
        score = self.data.iloc[idx]['score']
        inputs = self.tokenizer(
            query + " [SEP] " + document,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs['token_type_ids'].flatten(),
            'score': torch.tensor(score, dtype=torch.float)
        }


column_names = ['query', 'document', 'score']
df = pd.read_csv('../data/train/top100_train.csv', names=column_names)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
train_dataset = RankingDataset(train_df, tokenizer)
val_dataset = RankingDataset(val_df, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)


class RankingModel(nn.Module):
    def __init__(self):
        super(RankingModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.dense = nn.Linear(self.bert.config.hidden_size, 256)
        self.relu = nn.ReLU()
        self.score_classifier = nn.Linear(256, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.dense(pooled_output)
        pooled_output = self.relu(pooled_output)
        score = self.score_classifier(pooled_output)
        return score


train_losses = []
val_losses = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RankingModel().to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.MSELoss()
num_epochs = 15
accumulation_steps = 16
scaler = GradScaler()
early_stopping_patience = 3
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", leave=True)):
        with autocast():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            scores = batch['score'].to(device)
            predictions = model(input_ids, attention_mask, token_type_ids)
            loss = loss_fn(predictions.squeeze(), scores) / accumulation_steps

        scaler.scale(loss).backward()
        total_loss += loss.item() * accumulation_steps

        if (step + 1) % accumulation_steps == 0 or step + 1 == len(train_dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if (step + 1) % 100 == 0:
            print(f'Step {step + 1}, Current Loss: {loss.item()}')

    avg_train_loss = total_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            scores = batch['score'].to(device)
            predictions = model(input_ids, attention_mask, token_type_ids)
            loss = loss_fn(predictions.squeeze(), scores)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_dataloader)
    val_losses.append(avg_val_loss)

    print(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        print(f'No improvement for {patience_counter} epochs.')

    if patience_counter >= early_stopping_patience:
        print("Early stopping triggered.")
        break

model_path = "../data/bert"
tokenizer.save_pretrained(model_path)
torch.save(model.state_dict(), model_path + "/bert_weights.pth")

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid(True)
plt.savefig(f'{model_path}/training_validation_loss.png')
plt.show()
