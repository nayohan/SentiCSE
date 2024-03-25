# SentiCSE: A Sentiment-aware Contrastive Sentence Embedding Framework with Sentiment-guided Textual Similarity
Jaemin Kim*, Yohan Na*, Kangmin Kim, Sang Rak Lee, Dong-Kyu Chae

DILab, Hanyang University

**Our paper was published in LREC-COLING 2024.**

## Introduction
The overview of SentiCSE. In the sentence-level objective, we take two anchors, one for `positive' polarity and the other for `negative' polarity. We then encourage the sentiment representations to be more close to the corresponding sentences belonging to the same polarity, and to be far from the corresponding sentences associated with different polarities. In the word-level objective, our model tries to predict the masked words as in conventional MLM.

<img src="https://github.com/nayohan/SentiCSE/assets/54879393/c8e33eba-74f0-4caf-b9be-dfe094c45c1b" width="1000" />


## Create SgTS dataset
Comparison of STS and our SgTS. STS measures similarity of two sentences based on contextual semantics while SgTS judges similarity based on their sentiment polarities.

You can create the SgTS datsets. find download code in SentiCSE/utils/create_valid_ssts_dataset.ipynb

<img src="https://github.com/nayohan/SentiCSE/assets/54879393/b812f291-efc1-44f5-8f99-fdce8c3947d4" width="500" />



## Quick Start for Fine-tunning
Our experiments contain sentence-level sentiment classification (e.g. SST-2 / MR / IMDB / Yelp-2 / Amazon) 

### Use SentiCSE with Huggingface
You can also load our base model in huggingface ([https://huggingface.co/DILAB-HYU/SentiCSE](https://huggingface.co/DILAB-HYU/SentiCSE)):
```python
import torch
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, AutoModel


tokenizer = AutoTokenizer.from_pretrained("DILAB-HYU/SentiCSE")
model = AutoModel.from_pretrained("DILAB-HYU/SentiCSE")

# Tokenize input texts
texts = [
    "The food is delicious.",
    "The atmosphere of the restaurant is good.",
    "The food at the restaurant is devoid of flavor.",
    "The restaurant lacks a good ambiance."
]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Get the embeddings
with torch.no_grad():
    embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

# Calculate cosine similarities
# Cosine similarities are in [-1, 1]. Higher means more similar
cosine_sim_0_1 = 1 - cosine(embeddings[0], embeddings[1])
cosine_sim_0_2 = 1 - cosine(embeddings[0], embeddings[2])
cosine_sim_0_3 = 1 - cosine(embeddings[0], embeddings[3])

print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[1], cosine_sim_0_1))
print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[2], cosine_sim_0_2))
print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[3], cosine_sim_0_3))

```

## Getting Started
In the following section, we describe how to train a SentiCSE model by using our code.

### Requirements
```bash
conda create -n simcse python=3.8 -y
conda activate simcse
pip install transformers==4.2.1
pip install -r requirements.txt
```

### Pre-training SentiCSE
```bash
# run few-shot
run_few_shot_exp.sh

#run linear-probing
bash run_senticse_pretrain.sh
```

### Evaluation
Our evaluation code for sentence embeddings is based on a modified version of SentEval. It evaluates sentence embeddings on Sentiment-guided textual similarity (SgTS) tasks and downstream transfer tasks. For SgTS tasks, our evaluation takes the "all" setting, and report Spearman's correlation.

```bash
bash run_senticse_linear_probe.sh
```

```bash
+----------+----------------+-------+-------+--------+--------+-------+
| Model    | setting        | IMDB  | SST2  | Yelp-2 | Amazon | MR    |
+----------+----------------+-------+-------+--------+--------+-------+
|          | 1-shot         | 82.64 | 92.92 | 89.72  | 89.04  | 87.38 |
| SentiCSE | 5-shot         | 88.12 | 94.50 | 92.08  | 90.40  | 88.00 |
|          | linear-probing | 94.03 | 95.18 | 95.86  | 93.69  | 89.49 |
+----------+----------------+-------+-------+--------+--------+-------+
```

## Contributors
The main contributors of the work are: 
- [Jaemin Kim](https://github.com/kimfunn)\*
- [Yohan Na](https://github.com/nayohan)\*
- [Kangmin Kim](https://github.com/Gangsss)
- [Sangrak Lee](https://github.com/PangRAK)

\*: Equal Contribution

## Citation
Please cite the repo if you use the data or code in this repo.

```
@article{2024SentiCSE,
  title={SentiCSE: A Sentiment-aware Contrastive Sentence Embedding Framework with Sentiment-guided Textual Similarity},
  author={Kim, Jaemin and Na, Yohan and Kim, Kangmin and Lee, Sangrak and Chae, Dong-Kyu},
  journal={Proceedings of the 30th International Conference on Computational Linguistics (COLING)},
  year={2024},
}
```

