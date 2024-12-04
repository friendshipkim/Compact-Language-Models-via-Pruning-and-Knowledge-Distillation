"""
one-shot width pruning of LLAMA models
"""
import yaml
import argparse
from tqdm import tqdm

import torch
from datasets import load_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils_llama import prune
import pickle
device = (
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda:0" if torch.cuda.is_available() else "cpu")
)
# device = "cpu"
torch.manual_seed(1337)


def get_calib_data_iter(tokenizer, data="wikitext", batch_size=64, calib_size=512, max_sequence_length=512):
    if data == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
        text_column = "text"
    elif data == "cnn_dailymail":
        dataset = load_dataset("cnn_dailymail", name="3.0.0", split="train")
        text_column = "article"
    else:
        # Assume a local JSON dataset with a column named "text"
        dataset = load_dataset("json", data_files=data, split="train")
        text_column = "text"
    
    # Add text length column and sort to avoid padding
    def add_length(example):
        example['length'] = len(example[text_column])
        return example
    dataset = dataset.map(add_length)
    dataset = dataset.sort('length', reverse=True)
    
    calib_size = max(min(len(dataset), calib_size), batch_size)
    for i in range(calib_size // batch_size):
        batch = dataset[i * batch_size : (i + 1) * batch_size][text_column]
        tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_sequence_length)
        # add labels
        tokenized_batch["labels"] = tokenized_batch["input_ids"]
        yield tokenized_batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config YAML file')
    args = parser.parse_args()

    # Load YAML config and add config to args
    print(f"Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    for key, value in config.items():
        setattr(args, key, value)
    
    # load model and tokenizer
    print(f"Loading model from HF: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # if pad_token is not set, set it to eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer._pad_token = tokenizer.eos_token
        print(f"Set pad_token_id to {tokenizer.pad_token_id}")
        
    
    # prepare calibration data
    data_iter = get_calib_data_iter(
        tokenizer,
        args.dataset_name,
        args.test_batch_size,
        args.calib_size,
        args.max_seq_len,
    )
    # first_batch = next(data_iter)
    dataloader = [data for data in data_iter]
    
    # prune
    configurations = [
        [
            ("width_hidden", args.width_hidden),
            ("width_intermediate", args.width_intermediate),
            ("width_attn", args.width_attn),
            ("depth", args.depth),
        ]
    ]
    pruned_model, keep_idx_dict = prune(
        model,
        calibration_loader=dataloader,
        device=device,
        batch_agg_func=args.batch_agg_func,
        pruning_strategies=configurations,
    )
    
    # upload to HF hub
    # create repo name from base model and pruning ratios
    repo_name = f"{args.model_name.split('/')[-1]}-pruned-h{args.width_hidden}-i{args.width_intermediate}-a{args.width_attn}-d{args.depth}"
    # repo_name = repo_name.replace(".", "_")
    
    print(f"Uploading pruned model to HF Hub as: {repo_name}")
    
    # create repo
    
    pruned_model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)
    
    print("Upload complete!")
    
    # save keep_idx_dict
    with open(f"{repo_name}.pkl", "wb") as f:
        pickle.dump(keep_idx_dict, f)
    print(f"Saved keep_idx_dict to: {repo_name}.pkl")
