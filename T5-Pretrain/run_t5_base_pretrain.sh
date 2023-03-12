# target length is set to 512 in original script


python run_t5_mlm.py \
	--output_dir="./t5-large-pretrain-adamW" \
	--model_type="t5" \
	--config_name="t5-large" \
	--tokenizer_name="t5-large" \
	--train_file="pile-base.json" \
	--preprocessing_num_workers="30" \
	--validation_split_percentage="0" \
	--do_train \
	--overwrite_output_dir \
    --mlm_probability="0.15" \
    --mean_noise_span_length="3.0" \
	--per_device_train_batch_size="16" \
	--per_device_eval_batch_size="16" \
	--learning_rate="0.0001" \
	--warmup_steps="10000" \
    --max_steps="1000000000" \
	--logging_steps="10" \
	--save_steps="10000" \
	--fp16 \
