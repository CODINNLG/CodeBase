python run_summarization_no_trainer.py \
    --model_name_or_path t5-base \
    --dataset_name cnn_dailymail \
    --dataset_config_name "2.0.0" \
    --source_prefix "summarize: " \
    --preprocessing_num_workers 16 \
    --max_source_length 512 \
    --max_target_length 128 \
    --checkpointing_steps 5000 \
    --output_dir ./tst-summarization \
    --overwrite_output_dir \
    --do_train \
    --per_device_train_batch_size 16 \
    --num_warmup_step 10000 \
    --do_eval \
    --per_device_eval_batch_size 16 \
    --do_predict \
    --num_beams 4 \
    --length_penalty 0.6 \
    --learning_rate 0.00001 \
    --seed 42 \
    --max_train_steps 50 \

# accelerate launch run_summarization_no_trainer.py \
#     --model_name_or_path t5-small \
#     --dataset_name cnn_dailymail \
#     --dataset_config "3.0.0" \
#     --source_prefix "summarize: " \
#     --output_dir ~/tmp/tst-summarization