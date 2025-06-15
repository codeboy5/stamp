#!/bin/bash

rephrasing_model="meta-llama/Llama-3.1-8B-Instruct"

#* ARC-Challenge
dataset_name="arc_c"
seeds=(1009 3217 5083 7219 14153 15491 15661 23977 37213)
for seed in "${seeds[@]}"; do
    echo "Running for DATASET: ${dataset_name} and SEED: ${seed}"
    python src/rephrase_dataset.py --rephrasing_model $rephrasing_model --batch_size 16 \
    --dataset_name "files/${dataset_name}/original.json" \
    --text_field "question" \
    --watermark_seed $seed
done

# #* GSM8K
dataset_name="gsm8k"
seeds=(1283 1453 1871 2017 15991 18077 21269 39631 43313)
for seed in "${seeds[@]}"; do
    echo "Running for DATASET: ${dataset_name} and SEED: ${seed}"
    python src/rephrase_dataset.py --rephrasing_model $rephrasing_model --batch_size 16 \
    --dataset_name "files/${dataset_name}/original.json" \
    --text_field "question" \
    --watermark_seed $seed
done


# #* MMLU
dataset_name="mmlu"
seeds=(1069 1427 1787 2027 16963 17203 21929 42223 49409)
for seed in "${seeds[@]}"; do
    echo "Running for DATASET: ${dataset_name} and SEED: ${seed}"
    python src/rephrase_dataset.py --rephrasing_model $rephrasing_model --batch_size 16 \
    --dataset_name "files/${dataset_name}/original.json" \
    --text_field "question" \
    --watermark_seed $seed
done


# #* TriviaQA
dataset_name="trivia_qa"
seeds=(1697 1931 2357 10799 16427 24151 30307 32341 46807)
for seed in "${seeds[@]}"; do
    echo "Running for DATASET: ${dataset_name} and SEED: ${seed}"
    python src/rephrase_dataset.py --rephrasing_model $rephrasing_model --batch_size 16 \
    --dataset_name "files/${dataset_name}/original.json" \
    --text_field "question" \
    --watermark_seed $seed
done