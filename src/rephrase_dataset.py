import os
import json
import torch
import argparse

from tqdm import tqdm
from datasets import load_dataset
from transformers import LogitsProcessorList, AutoTokenizer, AutoModelForCausalLM, WatermarkingConfig, set_seed

from utils import get_benchmark_rephrase_prompt

def main(args):
    set_seed(seed=args.watermark_seed)

    model_name = args.rephrasing_model

    # Load the rephrasing_model
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print(f"Successfully loaded: {model_name}")

    # Get the watermarking args
    watermarking_config = WatermarkingConfig(greenlist_ratio=args.gamma, bias=args.delta, seeding_scheme="lefthash", hashing_key=args.watermark_seed)

    dataset = load_dataset("json", data_files=args.dataset_name, split="train")
    dataset = dataset.map( get_benchmark_rephrase_prompt, fn_kwargs={"tokenizer": tokenizer, "text_field": args.text_field}, load_from_cache_file=False)

    if "trivia_qa" in args.dataset_name:
        dataset = dataset.remove_columns(['entity_pages', 'search_results', 'answer'])
    elif "arc_c" in args.dataset_name:
        dataset = dataset.remove_columns(['choices'])

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)

    #* the sampling arguments
    generation_args = { "do_sample": True, 
                       "temperature": 1.0, 
                       "pad_token_id": tokenizer.pad_token_id, 
                       "max_new_tokens": 512,
                        "watermarking_config": watermarking_config }
    
    #* Generate and store the rephrased versions
    rephrased_questions = []
    for batch in tqdm(dataloader):
        prompt = batch["rephrase_prompt"]
        tokenized = tokenizer(prompt, padding=True, return_tensors="pt", add_special_tokens=False)
        n_input_tokens = tokenized["input_ids"].shape[1]

        outputs = model.generate(**tokenized.to(model.device), **generation_args)
        response = tokenizer.batch_decode(outputs[:, n_input_tokens:], skip_special_tokens=True)
        rephrased_questions.extend(response)
    
    results = []
    for i in range(len(dataset)):
        results.append( json.dumps({ **dataset[i], "rephrased_question": rephrased_questions[i].strip() }) )

    #* We store in the same folder as the input directory
    save_dir = os.path.join(os.path.dirname(args.dataset_name), f"rephrased_gamma{args.gamma}_delta{args.delta}_k1" )
    os.makedirs(save_dir, exist_ok=True)
    full_path = f"{save_dir}/watermarked_seed{args.watermark_seed}.json"
    with open(full_path, "w") as f:
        print(f"Writing output to {full_path}")
        f.write("\n".join(results))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--rephrasing_model", type=str, default="meta-llama/Llama-3.1-70B-Instruct",
                        help="The model to use for rephrasing")
    parser.add_argument("--batch_size", type=int, default=8, help="The batch size to use for rephrasing" )

    #* The dataset to rephrase
    parser.add_argument("--dataset_name", type=str, required=True, help="The dataset to rephrase, local or hosted on HF")
    parser.add_argument("--from_hf", type=int, default=0, help="If set, will load the dataset from huggingface")
    parser.add_argument("--text_field", type=str, default="text", help="The text field to use for rephrasing")

    #* Set the parameters for watermarking
    parser.add_argument("--watermark_seed", type=int, required=True)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--delta", type=float, default=1.0)

    args = parser.parse_args()

    main(args)