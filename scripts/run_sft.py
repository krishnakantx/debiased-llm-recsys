#!/usr/bin/env python3
"""Run SFT training only."""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from data.movielens import MovieLensDataset
from src.preference_extraction import PreferenceExtractor
from src.preference_pairs import PreferencePairConstructor
from src.sft_trainer import SFTTrainer, SFTConfig


def main():
    parser = argparse.ArgumentParser(description="Run SFT training")
    parser.add_argument("--config", "-c", default="configs/test_tiny.yaml")
    parser.add_argument("--max-samples", "-n", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    print("=" * 50)
    print("SFT Training")
    print("=" * 50)
    
    print("\nLoading dataset...")
    dataset = MovieLensDataset()
    extractor = PreferenceExtractor(dataset)
    constructor = PreferencePairConstructor(dataset)
    
    user_ids = dataset.get_all_users()
    if args.max_samples:
        user_ids = user_ids[:args.max_samples]
    
    print(f"Generating SFT data for {len(user_ids)} users...")
    sft_data = constructor.generate_sft_dataset(extractor, user_ids)
    
    split = int(len(sft_data) * 0.9)
    train_data, val_data = sft_data[:split], sft_data[split:]
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    sft_cfg = SFTConfig(
        model_name=config.get('model_name'), max_seq_length=config.get('max_seq_length', 256),
        lora_r=config.get('lora_r', 16), lora_alpha=config.get('lora_alpha', 32),
        lora_dropout=config.get('lora_dropout', 0.05), target_modules=config.get('target_modules', ['q_proj', 'v_proj']),
        **config.get('sft', {})
    )
    
    trainer = SFTTrainer(sft_cfg)
    
    if args.dry_run:
        print("\nDry run - loading model only...")
        trainer.load_model()
        print("Model loaded successfully!")
        sample = train_data[0]['prompt']
        print(f"\nTest prompt: {sample[:80]}...")
        print(f"Generated: {trainer.generate(sample, max_new_tokens=50)[:100]}...")
    else:
        trainer.train(train_data, val_data)
        print(f"\nModel saved to: {sft_cfg.output_dir}")


if __name__ == "__main__":
    main()
