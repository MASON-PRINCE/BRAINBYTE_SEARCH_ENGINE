import os

import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import pandas as pd
from retrieval_model.bm25 import sample_retrieve_function, sample_get_document, get_document
from retrieval_model.evaluation import calculate_n_dcg, average_precision
import string
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from retrieval_model.bm25 import retrieve_function


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


tokenizer = BertTokenizer.from_pretrained("../data/bert")

model = RankingModel()
model.load_state_dict(torch.load("../data/bert/bert_weights.pth"))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def find_synonyms(word):
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    synonyms.add(word)
    return list(synonyms)


stop_words = set(stopwords.words('english'))


def predict_function(query, test=True):
    query_no_punctuation = query.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(query_no_punctuation)
    filtered_query = [word for word in words if not word.lower() in stop_words]
    keywords_synonyms = []
    for word in filtered_query:
        syn = find_synonyms(word)
        keywords_synonyms.extend(syn)
    extended_query = ' '.join(keywords_synonyms)

    if not test:
        documents = retrieve_function(query)
    else:
        documents = sample_retrieve_function(extended_query)

    max_length = 512
    input_ids_list = []
    attention_mask_list = []
    token_type_ids_list = []
    scores = []

    for document in documents:
        (_, value), = document.items()
        inputs = tokenizer(
            query + " [SEP] " + str(np.log1p(value[4])) + " [SEP] " + value[3],
            padding='max_length',
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )
        input_ids_list.append(inputs['input_ids'].to(device))
        attention_mask_list.append(inputs['attention_mask'].to(device))
        token_type_ids_list.append(inputs['token_type_ids'].to(device))

    with torch.no_grad():
        for i in range(len(documents)):
            predictions = model(input_ids_list[i], attention_mask_list[i], token_type_ids_list[i])
            scores.append(predictions.squeeze().item())

    doc_scores = list(zip(documents, scores))
    doc_scores_sorted = sorted(doc_scores, key=lambda x: x[1], reverse=True)[:100]

    if test:
        t100_doc, _ = zip(*doc_scores_sorted)
        t100_keys = [list(doc.keys())[0] for doc in t100_doc]
        id_scores = list(zip(t100_keys, scores))
        columns_name = ['qid', 'query']
        df = pd.read_csv('../data/sample_data/sample_queries.tsv', delimiter='\t', names=columns_name)
        qid = df.loc[df['query'] == query, 'qid'].iloc[0]
        ground_truth_scores = []
        predict_scores = []
        with open('../data/sample_data/sample_top100.txt', 'r', encoding='utf-8') as file:
            for line in file:
                t100 = line.split()
                if t100[0] == str(qid):
                    ground_truth_scores.append(float(t100[4]))
                    predict_score = float(next((score for doc_id, score in id_scores if doc_id == t100[2]), 0))
                    predict_scores.append(predict_score)
        precision = len(predict_scores) - predict_scores.count(0.0)
        n_dcg = calculate_n_dcg(ground_truth_scores, predict_scores)
        ap = average_precision(ground_truth_scores, predict_scores)

        print(f'Precision@100: {precision}')
        print(f'nDCG@100: {n_dcg:.2f}')
        print(f'MAP@100: {ap:.2f}')

        '''
        results = {
            "Precision@100": [precision],
            "nDCG@100": [n_dcg],
            "MAP@100": [ap]
        }

        results_df = pd.DataFrame(results)
        file_path = 'evaluation_metrics_bert@10.csv'
        if not os.path.isfile(file_path):
            results_df.to_csv(file_path, index=False)
        else:
            results_df.to_csv(file_path, mode='a', header=False, index=False)
        '''

    doc_info = []
    bert_rank = 1
    for doc, score in doc_scores_sorted:
        (key, value), = doc.items()
        if not test:
            document = get_document(key)
        else:
            document = sample_get_document(key[15:])
        doc_info.append({document.get('docid'): [document.get('title'), document.get('url'),
                                                 document.get('body')[:150]]})
        print(f"Document: {key}, Score: {score}, BM25: {value[2]}, Bert: rank{bert_rank}")
        bert_rank += 1

    return doc_info


'''
file_path = '../data/sample_data/sample_queries.tsv'
columns_name = ['qid', 'query']
df = pd.read_csv(file_path, delimiter='\t', names=columns_name)
for index, row in df.iterrows():
    query = row['query']
    predict_function(query)
'''
