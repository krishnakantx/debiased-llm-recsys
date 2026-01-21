#!/usr/bin/env python3
"""Run DPO training only."""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from data.movielens import MovieLensDataset
from src.preference_extraction import PreferenceExtractor
from src.preference_pairs import PreferencePairConstructor
from src.dpo_trainer import DPORecommendationTrainer, DPOTrainerConfig


def main():
    parser = argparse.ArgumentParser(description="Run DPO training")
    parser.add_argument("--config", "-c", default="configs/test_tiny.yaml")
    parser.add_argument("--sft-path", default=None)
    parser.add_argument("--max-samples", "-n", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    print("=" * 50)
    print("DPO Training")
    print("=" * 50)
    
    print("\nLoading dataset...")
    dataset = MovieLensDataset()
    extractor = PreferenceExtractor(dataset)
    constructor = PreferencePairConstructor(dataset)
    
    user_ids = dataset.get_all_users()
    if args.max_samples:
        user_ids = user_ids[:args.max_samples]
    
    print(f"Generating DPO pairs for {len(user_ids)} users...")
    dpo_data = constructor.generate_dpo_dataset(extractor, user_ids)
    
    split = int(len(dpo_data) * 0.9)
    train_data, val_data = dpo_data[:split], dpo_data[split:]
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    sft_path = args.sft_path or config.get('sft', {}).get('output_dir', './outputs/sft')
    
    dpo_cfg = DPOTrainerConfig(
        model_name=config.get('model_name'), sft_model_path=sft_path if Path(sft_path).exists() else None,
        max_seq_length=config.get('max_seq_length', 256), lora_r=config.get('lora_r', 16),
        lora_alpha=config.get('lora_alpha', 32), lora_dropout=config.get('lora_dropout', 0.05),
        target_modules=config.get('target_modules', ['q_proj', 'v_proj']), **config.get('dpo', {})
    )
    
    trainer = DPORecommendationTrainer(dpo_cfg)
    
    if args.dry_run:
        print("\nDry run - loading models only...")
        trainer.load_models()
        print("Models loaded successfully!")
    else:
        trainer.train(train_data, val_data)
        print(f"\nModel saved to: {dpo_cfg.output_dir}")


if __name__ == "__main__":
    main()
