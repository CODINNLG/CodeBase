Dataset_dir=/opt/data/private/data/roc/segment/json
Model_architecture='roberta-base'
Output_dir='./output_ckp'

mkdir $Output_dir
# export PYTHONPATH=/opt/data/private/transformers/examples/MyProject/non_autoregressive_2
export CUDA_VISIBLE_DEVICES=0

python train.py \
    --dataset_dir $Dataset_dir \
    --train_file train.json \
    --validation_file val.json \
    --max_source_length 512 \
    --preprocessing_num_workers 1 \
    --overwrite_cache True \
    --max_target_length 512 \
    --pad_to_max_length \
    --model_name_or_path $Model_architecture \
    --config_name $Model_architecture \
    --tokenizer_name $Model_architecture \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 100 \
    --gradient_accumulation_steps 4 \
    --num_warmup_steps 500 \
    --output_dir $Output_dir \
    --sim_alpha 0.0 \
    --pos_alpha 0.0 \
    --learning_rate 5e-6







