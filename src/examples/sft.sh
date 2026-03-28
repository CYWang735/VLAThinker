export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
if [ -d "$OUTPUT_DIR/wandb" ]; then
    rm -rf $OUTPUT_DIR/wandb
    echo "Wandb dir clean"
else
    echo "Wandb dir doesn't exists"
fi

export WANDB_PROJECT="VLA-Thinker"
export WANDB_MODE="online"
export WANDB_NAME="libero_cot"
export TOKENIZERS_PARALLELISM="false"
export TRANSFORMERS_NO_ADVISORY_WARNINGS="true"
# export CUDA_VISIBLE_DEVICES=0,1,2,3


if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

# For run in a single node/machine
# accelerate launch main.py \
#     --deepspeed="./configs/zero2.json" \
#     ...

deepspeed src/train.py \
    --deepspeed ./src/configs/zero2.json \
    --base_model_path Base_model_path \
    --num_images_in_input 2 \
    --run_name $WANDB_NAME \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size 8 \
    --auto_find_batch_size false \
    --max_steps 150005 \
    --num_train_epochs 100 \
    --save_steps 10000 \
    --save_total_limit 40 \
    --warmup_steps 1000 \
    --lr_scheduler_type cosine \
    --learning_rate 2.5e-5 \
    --vision_lr 5e-6 \
    --merger_lr 2.5e-6 \
    --weight_decay 1e-10 \
    --optim adamw_torch \
    --dataloader_num_workers 32 \
    --report_to wandb \
    --logging_steps 1 \
    --log_level info \
    --seed 429 \
    --resume false \
    --gradient_accumulation_steps 2 \
    
