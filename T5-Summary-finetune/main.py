from transformers import (
    Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq,
    T5ForConditionalGeneration, T5Tokenizer,
    set_seed,
)
import argparse
import torch
from torch.utils.data import Dataset
import pandas as pd
from datasets import load_dataset, load_from_disk
import os
from rouge_score import rouge_scorer
import numpy as np
import nltk
import evaluate


parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=1)
parser.add_argument("--model_name_or_path", default="T5-base")
parser.add_argument("--data_path", default="news_summary.csv")
parser.add_argument("--source_max_length", default=512)
parser.add_argument("--target_max_length", default=50)
parser.add_argument("--output_dir", default='./temp-output')
args = parser.parse_args()
set_seed(args.seed)


model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
metric = evaluate.load("rouge")

def process_data(data):
    for i in range(len(data['headlines'])):
        data["headlines"][i] = "Summary: " + data["headlines"][i]
    # print(data['text'])
    model_inputs = tokenizer(
        data['text'], 
        max_length=args.source_max_length, 
        truncation=True, padding="max_length", 
        return_tensors="pt"
    )
    target_tokenized = tokenizer(
        data['headlines'], 
        max_length=args.target_max_length, 
        truncation=True, padding="max_length", 
        return_tensors="pt"
    )
    model_inputs['labels'] = target_tokenized["input_ids"]
    return model_inputs
    
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    
    return preds, labels

def compute_rouge(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result

raw_dataset = load_dataset('csv', data_files='news_summary.csv', split="train")
column_names = raw_dataset.column_names
tokenized_dataset = raw_dataset.map(process_data, remove_columns=column_names, num_proc=4, batched=True)
train_dataset, test_dataset = tokenized_dataset.train_test_split(train_size=0.99, seed=args.seed).values()
train_dataset.save_to_disk("encoded_dataset/train.hf")
test_dataset.save_to_disk("encoded_dataset/test.hf")

# train_dataset = load_from_disk("encoded_dataset/train.hf")
# test_dataset = load_from_disk("encoded_dataset/test.hf")
# data_collator=DataCollatorForSeq2Seq(tokenizer, model)

training_args = Seq2SeqTrainingArguments(
    output_dir='./temp-output', 
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=1e-4,
    save_strategy='no',
    predict_with_generate=True,
    logging_steps=10,
)

trainer = Seq2SeqTrainer(
    model=model, 
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model),
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_rouge,
)

trainer.train()
trainer.save_model()

predict_results = trainer.predict(
    test_dataset, metric_key_prefix="predict", 
    max_length=150, num_beams=2, repetition_penalty=2.5, length_penalty=1.0, early_stopping=True)
