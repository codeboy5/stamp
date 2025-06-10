import os
import json
import torch
import argparse

from metrics import aggregate_metrics
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

def prepare_model(model_name, quant=torch.bfloat16):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = 1024
        tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, 
                                        torch_dtype=quant, device_map="auto")
    return model, tokenizer

def main(args):

    print(f"Processing Model: {args.model_name}")
    model, tokenizer = prepare_model(args.model_name)

    print(f"Processing Dataset: {args.dataset_name}")
    dataset = load_dataset("json", data_files=args.dataset_name, split="train")
    
    METRIC_LIST = args.metrics
    TEXT_COLUMNS = args.text_columns

    dataset_metrics = {}

    for column_name in TEXT_COLUMNS:
        col_metrics = aggregate_metrics(model, tokenizer, dataset, METRIC_LIST, 
                                    args=None, text_field=column_name, batch_size=args.batch_size)
        dataset_metrics[column_name] = col_metrics

    os.makedirs( os.path.dirname(args.output_path) , exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(dataset_metrics, f, indent=4)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True, help="The name of the model to use")
    parser.add_argument("--dataset_name", type=str, required=True, help='The name of the dataset to use')
    parser.add_argument("--batch_size", type=int, default=16, help="The batch size to use")
    parser.add_argument("--output_path", type=str, required=True, help="The path to save the computed metrics")

    parser.add_argument("--metrics", nargs='+', default=["k_min_probs", "ppl", "zlib_ratio", "k_max_probs"], 
                        help="List of metrics to compute")
    parser.add_argument("--text_columns", nargs='+', default=["full_text"], 
                        help="List of text columns to process")

    args = parser.parse_args()

    main(args)