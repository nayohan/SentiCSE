import os
import io
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from datasets import load_dataset, load_metric, DatasetDict, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,TrainingArguments,Trainer
from transformers import EarlyStoppingCallback, set_seed

import wandb
import argparse

def preprocess(dataset):
    t = dataset['text']
    t = '@user' if t.startswith('@') and len(t) > 1 else t
    t = 'http' if t.startswith('http') else t
    dataset['text'] = t
    return dataset

def fix_seed(seed):
    # need to args
    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def loadFile(fpath):
    sst_data = {'idx': [], 'text': [], 'label': []}
    with io.open(fpath, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            sample = line.strip().split('\t')
            sst_data['idx'].append(int(idx))
            sst_data['label'].append(int(sample[1]))
            sst_data['text'].append(sample[0])

    assert max(sst_data['label']) == 1
    return sst_data

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset_name", default="rotten_tomatoes",  type=str,  help="which dataset you train")
parser.add_argument("-m", "--model_name", default="roberta-basse",  type=str,  help="which model you train")
parser.add_argument("-t_num", "--train_num", default=10,  type=int, help="train dataset num for few-shot learning")
parser.add_argument("-v_num", "--valid_num", default=10,  type=int, help="valid dataset num for few-shot learning")
parser.add_argument("-seed", default=0,  type=int, help="which seed you use")
parser.add_argument("-gpu", "--gpu_number", default=0, type=int, help="which gpu you use")
parser.add_argument("-run", "--seed_run", default=3, type=int, help="seed run iteration")
parser.add_argument("-e_step", default=1, type=int, help='eval_step')
parser.add_argument("-lr", default=1e-5, type=float, help='learning_rate')
parser.add_argument("-bs", default=1, type=int, help='batch_size')
parser.add_argument("-epoch", default=1, type=int, help="epoch to run")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_number)

wandb.init(project="sentiment-one-shot-final-please",
            #name=f'{args.dataset_name}_{args.model_name}_num{args.train_num}_seed{args.seed}',
            name=f'{str(args.model_name).split("/")[-1]}{args.dataset_name}_epoch_{args.epoch}_bs{args.bs}lr_{args.lr}_eval_step{args.e_step}',
            config=args,
            #tags=[args.model_name, args.train_num, args.dataset_name],
            #tags=["eval_step20", "patience5", "frozen"]
            )

eval_accuracy_list = []
for seed in range(1,args.seed_run+1):
    args.seed = seed
    fix_seed(args.seed)

    dataset = load_dataset(args.dataset_name)
    train_strat, valid_strat = train_test_split(dataset['train'], train_size=args.train_num, random_state=args.seed, stratify=dataset['train']['label'])
    train = Dataset.from_pandas(pd.DataFrame(train_strat))
    valid = Dataset.from_pandas(pd.DataFrame(valid_strat)).shuffle(seed=0).select(range(args.valid_num))
    
    if args.dataset_name=="sst2": # 
        train = train.rename_column("sentence", "text")
        valid = valid.rename_column("sentence", "text")
        test = Dataset.from_dict(loadFile(os.path.join('/home/uj-user/SenCSE/SentEval/data/downstream/SST/binary', 'sentiment-test')))
    else:
        test = load_dataset(args.dataset_name)['test']
        
    test = test.shuffle(seed=0).select(range(args.valid_num))
    
    dataset = DatasetDict({'train':train, 'validation':valid, 'test':test})
    print(dataset['train'])

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2).to("cuda")
    # for name, param in model.electra.named_parameters():
    #     param.requires_grad = False
        
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized_stance = dataset.map(preprocess_function, batched=True)
    print(tokenized_stance)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    wandb.watch(model)

    os.makedirs('./result_sentiment', exist_ok=True)
    training_args = TrainingArguments(
        output_dir=f'./result_sentiment/{args.dataset_name}_{args.model_name}_num{args.train_num}',
        learning_rate=args.lr,
        evaluation_strategy = "steps",
        eval_steps=args.e_step,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=100,
        num_train_epochs=args.epoch,
        weight_decay=0.01,
        load_best_model_at_end = True,
        save_total_limit=1,
        fp16=True,
    )

    metric = load_metric("accuracy")
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=-1)
        results = metric.compute(predictions=predictions, references=labels)
        return results

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_stance["train"],#.shuffle(seed=args.seed).select(range(args.train_num)),
        eval_dataset=tokenized_stance["validation"],#.shuffle(seed=0).select(range(500)),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics, 
        callbacks = [EarlyStoppingCallback(early_stopping_patience=5)],
        )

    trainer.train()
    result = trainer.evaluate(eval_dataset=tokenized_stance["test"])
    print(result)
    wandb.log(result)
    eval_accuracy_list.append(result['eval_accuracy'])
    top3_list = sorted(eval_accuracy_list, reverse=True)[:3]

wandb.log({'avg_eval_acc': sum(eval_accuracy_list) / len(eval_accuracy_list), 
           'top3_avg_acc': sum(top3_list)/3,
           'top_acc': top3_list[0],
           })