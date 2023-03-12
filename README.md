# Run1: Pretrain T5-base (220M Parameters) from scratch on pile
model=T5-base, batch_size=8\*16=128, block_size=512, GPU-Mem=19581MiB\*1+19581MiB\*7

T5 details: 
- Pre-train each model for $2^{19}$ = **524,288** steps.
- A maximum sequence length of 512 and a batch size of 128 sequences. In total, this batch size and number of steps corresponds to pre-training on $2^{35}$ $\approx$ 34B tokens
- Note that $2^{35}$ tokens only covers a fraction of the entire C4 data set, so we never repeat any data during pre-training. (repeating data is detrimental.)

Time efficiency for training T5-base (220M Parameters) in 34B Tokens in 8\*A100:
- For T5-base, 1.5s/step, $\text{Time consumed} \approx 1.5 * 2^{19} = 786432s \approx 10 \text{days}$
- For T5-large, GPU OOM without deepspeed

# Run2: Finetune T5-base (220M Parameters) on CNN_DailyMail

results (reported in T5 Paoer): 42.05/20.34/39.40

results (implementations): 43.76/21.10/30.824

```bash
python run_summarization.py \
    --model_name_or_path t5-base \
    --dataset_name cnn_dailymail \
    --dataset_config "2.0.0" \
    --source_prefix "summarize: " \
    --preprocessing_num_workers 16 \
    --max_source_length 512 \
    --max_target_length 128 \
    --logging_strategy steps \
    --logging_steps 50 \
    --evaluation_strategy steps \
    --eval_steps 5000 \
    --save_strategy steps \
    --save_steps 5000 \
    --save_total_limit 4 \
    --metric_for_best_model "rouge2" \
    --output_dir ./tst-summarization \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --do_predict \
    --max_steps 262144 \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=4 \
    --warmup_steps 10000 \
    --predict_with_generate \
    --num_beams 4 \
    --learning_rate 0.00001 \
    --seed 42 \
```

# To do

- `length_penalty` -> `rouge-L`
- `adafactor` optimizer and `noam` schedule in pre-train and fine-tune.
    - compare with AdamW
    - the consistency of optimizer in pretrain and finetune
- \+ deepspeed
