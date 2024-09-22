# BRAINBYTE_SEARCH_ENGINE
## Introduction
A free, open-source, and offline research engine based on `BM25` and `BERT` models. 
Built using `PyTorch` and `PyQt6`.

## Usage
If you want to download the complete project, please click the link below:

[Download Complete Project](https://qmulprod-my.sharepoint.com/:f:/g/personal/jp2019213542_qmul_ac_uk/EhcpG7ny4jtCku95pvM1nAwB2zVuG9wRXK4Uk0QOylX2sw)

**Note that you can only input queries listed in `/data/sample_data/sample_queries.tsv`.Because the complete dataset is too large, only test dataset was provided**

**If you only want to run the UI, you don't need to download `/data/train/top100_train.csv`.**

### Conda environment

First, install the Python dependencies:
```
conda create -n BRAINBYTE_SEARCH_ENGINE python==3.9
conda activate BRAINBYTE_SEARCH_ENGINE
pip install -r environment.txt
```
### Setting `PYTHONPATH` in Conda Environment
Setting the `PYTHONPATH` environment variable after activating your environment allows Python to find and import modules from directories not installed in the standard library or virtual environment paths.

#### For Unix/Linux/macOS:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/your/module"
```
#### For Windows:
```cmd
set PYTHONPATH=%PYTHONPATH%;C:\path\to\your\module
```

### To run the UI:
```bash
python search_enging_UI.py
```
