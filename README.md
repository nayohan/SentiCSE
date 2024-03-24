# SentiCSE
## SentiCSE: A Sentiment-aware Contrastive Sentence Embedding Framework with Sentiment-guided Textual Similarity
Jaemin Kim*, Yohan Na*, Kangmin Kim, Sang Rak Lee, Dong-Kyu Chae

Hanyang University

## Create SgTS datset
![image](https://github.com/nayohan/SentiCSE/assets/54879393/6de74feb-b90e-408f-a4e4-a020dbda5060)


## Quick Start for Fine-tunning
Our experiments contain sentence-level sentiment classification (e.g. SST-2 / MR / IMDB / Yelp-2 / Amazon) 

### Load our model(base)
You can also load our base model in huggingface ([https://huggingface.co/nayohan/~](https://huggingface.co/nayohan/~)):
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("nayohan/")
model = AutoModelForSequenceClassification.from_pretrained("nayohan/")
```

## Getting Started

```bash
conda create -n simcse python=3.8 -y
conda activate simcse
pip install transformers==4.2.1
pip install -r requirements.txt
```

### Train SentiCSE
```bash
bash run_senticse_pretrain.sh
bash run_senticse_linear_probe.sh
```
