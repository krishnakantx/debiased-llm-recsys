#!/usr/bin/env python3
"""Evaluate a trained model."""

import argparse
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from data.movielens import MovieLensDataset
from src.preference_extraction import PreferenceExtractor
from src.evaluation import Evaluator


def parse_recommendations(text: str, dataset: MovieLensDataset) -> list:
    items = dataset.load_items()
    title_to_id = {row['title']: row['item_id'] for _, row in items.iterrows()}
    
    item_ids = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        if line in title_to_id:
            item_ids.append(title_to_id[line])
            continue
        for title, iid in title_to_id.items():
            if line.lower() in title.lower() or title.lower() in line.lower():
                item_ids.append(iid)
                break
    return item_ids


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--config", "-c", default="configs/test_tiny.yaml")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--num-users", "-n", type=int, default=100)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", "-o", default="eval_results.json")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    print("=" * 50)
    print("Model Evaluation")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    print(f"\nLoading model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    dtype = torch.float16 if device != "cpu" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(config.get('model_name'), torch_dtype=dtype, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, args.model_path).to(device)
    model.eval()
    
    print("\nLoading dataset...")
    dataset = MovieLensDataset()
    extractor = PreferenceExtractor(dataset)
    evaluator = Evaluator(dataset)
    
    _, test_df = dataset.get_train_test_split()
    test_users = test_df['user_id'].unique().tolist()[:args.num_users]
    print(f"\nEvaluating on {len(test_users)} users...")
    
    recommendations, ground_truth = {}, {}
    
    for uid in tqdm(test_users):
        user_test = test_df[test_df['user_id'] == uid]
        relevant = set(user_test[user_test['rating'] >= 4]['item_id'].tolist())
        if not relevant:
            continue
        
        ground_truth[uid] = relevant
        prompt = f"### Instruction:\n{extractor.format_prompt(uid)}\n\n### Response:\n"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True, pad_token_id=tokenizer.pad_token_id)
        
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        if "### Response:" in text:
            text = text.split("### Response:")[-1].strip()
        
        recommendations[uid] = parse_recommendations(text, dataset)
    
    print("\nComputing metrics...")
    results = evaluator.evaluate_all(recommendations, ground_truth, k=args.k)
    print("\n" + evaluator.format_results(results, k=args.k))
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
