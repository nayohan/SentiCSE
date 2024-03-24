# SentiCSE
## For paper: SentiCSE: A Sentiment-aware Contrastive Sentence Embedding Framework with Sentiment-guided Textual Similarity
Jaemin Kim*, Yohan Na*, Kangmin Kim, Sang Rak Lee, Dong-Kyu Chae

Hanyang University

## Getting Started

```bash
# for SimCSE
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
