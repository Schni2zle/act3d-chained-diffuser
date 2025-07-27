root=/tmp/rlbench
data_dir=$root/datasets/raw
output_dir=$root/datasets/packaged
task=stack_blocks_edit
train_dir="${task}_task_train"
val_dir="${task}_task_val"
train_episodes_per_task=1
val_episodes_per_task=1
image_size="256,256"


processes=3
python data_preprocessing/dataset_generator.py \
    --save_path=$data_dir/$train_dir \
    --tasks=$task \
    --image_size=$image_size \
    --renderer=opengl \
    --episodes_per_task=$train_episodes_per_task \
    --variations=1 \
    --offset=0 \
    --processes=$processes

python data_preprocessing/dataset_generator.py \
    --save_path=$data_dir/$val_dir \
    --tasks=$task \
    --image_size=$image_size \
    --renderer=opengl \
    --episodes_per_task=$val_episodes_per_task \
    --variations=1 \
    --offset=0 \
    --processes=$processes

for split_dir in $train_dir $val_dir; do
    python -m data_preprocessing.data_gen \
        --data_dir=$data_dir/$split_dir \
        --output=$output_dir/$split_dir \
        --image_size=$image_size \
        --max_variations=1 \
        --tasks=$task
done