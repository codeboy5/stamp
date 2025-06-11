

def get_watermark_args(args, tokenizer, device="cuda"):
    watermark_args = {
        "hash_key": args.watermark_seed,
        "gamma": args.gamma,
        "delta": args.delta,
        "seeding_scheme": args.seeding_scheme,
        "vocab": tokenizer.get_vocab().values(),
        "device": device,
    }

def get_benchmark_rephrase_prompt(example, tokenizer, text_field):

    messages = [
        {"role": "system", "content": "You are a bot that helps in rephrasing."},
        {"role": "user", "content": f"""Rephrase the question given below. Ensure you keep all details present in the original, without omitting anything or adding any extra information not present in the original question.\nQuestion: {example[text_field]}\nYour response should end with \"Rephrased Question: [rephrased question]\" """},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt += "Rephrased Question:"
    example['rephrase_prompt'] = prompt
    return example