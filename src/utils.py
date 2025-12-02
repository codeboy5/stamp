import numpy as np
from scipy import stats

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


# ---------------------------------------------------------------------------- #
#                                    STAMP                                     #
# ---------------------------------------------------------------------------- #


#! remove the outliers from the list of the calculated metrics
def remove_outliers(metrics, remove_frac=0.05, outliers = "zero"):
    # Sort the array to work with ordered data
    sorted_ids = np.argsort(metrics)
    
    # Calculate the number of elements to remove from each side
    total_elements = len(metrics)
    elements_to_remove_each_side = int(total_elements * remove_frac / 2) 
    
    # Ensure we're not attempting to remove more elements than are present
    if elements_to_remove_each_side * 2 > total_elements:
        raise ValueError("remove_frac is too large, resulting in no elements left.")
    
    # Change the removed metrics to 0.
    lowest_ids = sorted_ids[:elements_to_remove_each_side]
    highest_ids = sorted_ids[-elements_to_remove_each_side:]
    all_ids = np.concatenate((lowest_ids, highest_ids))

    # import pdb; pdb.set_trace()
    
    trimmed_metrics = np.copy(metrics)
    
    if outliers == "zero":
        trimmed_metrics[all_ids] = 0
    elif outliers == "mean" or outliers == "mean+p-value":
        trimmed_metrics[all_ids] = np.mean(trimmed_metrics)
    elif outliers == "clip":
        highest_val_permissible = trimmed_metrics[highest_ids[0]]
        lowest_val_permissible = trimmed_metrics[lowest_ids[-1]]
        trimmed_metrics[highest_ids] =  highest_val_permissible
        trimmed_metrics[lowest_ids] =   lowest_val_permissible
    elif outliers == "randomize":
        #this will randomize the order of metrics
        trimmed_metrics = np.delete(trimmed_metrics, all_ids)
    else:
        assert outliers in ["keep", "p-value"]
        pass
        
    return trimmed_metrics

#! detect contamination
def detect_contamination(computed_data, dataset_name, DATASET_CONSTANTS, subset_ids=None, private_key_ids=None, use_paired_test=True):

    #* get the perplexities for the publicaly released version of the dataset
    public_key = DATASET_CONSTANTS[dataset_name]['public_key']
    public_ppls = np.array( computed_data[dataset_name][f"{public_key}"]['ppl'] )
    if subset_ids is not None:
        public_ppls = public_ppls[subset_ids]
    public_ppls = remove_outliers( public_ppls, remove_frac=0.05, outliers="clip") #* remove outliers, basically 2.5% from each side

    #* get the perplexities for the confidential versions of the dataset
    all_private_ppls = []
    private_keys = np.array(DATASET_CONSTANTS[dataset_name]['private_keys'])
    if private_key_ids is not None:
        private_keys = private_keys[private_key_ids]

    # for private_key in DATASET_CONSTANTS[dataset_name]['private_keys']:
    for private_key in private_keys:
        private_ppls = np.array( computed_data[dataset_name][f"{private_key}"]['ppl'] )
        if subset_ids is not None:
            private_ppls = private_ppls[subset_ids]
        private_ppls = remove_outliers( private_ppls, remove_frac=0.05, outliers="clip")
        all_private_ppls.append( private_ppls )
    
    all_private_ppls = np.array(all_private_ppls)
    mean_private_ppls = np.mean(all_private_ppls, axis=0)

    if use_paired_test:
        test_statistic = stats.ttest_rel(public_ppls, mean_private_ppls, alternative="less")
    else:
        test_statistic = stats.ttest_ind(public_ppls, mean_private_ppls, alternative="less")

    return test_statistic.pvalue, np.log(test_statistic.pvalue)