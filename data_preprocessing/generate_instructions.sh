python data_preprocessing/preprocess_instructions.py \
 --tasks stack_blocks \
 --output output/instructions.pkl \
 --encoder clip \
 --batch 10 \
 --model_max_length 53 \
 --device cuda \
 --variations 0 \
 --verbose