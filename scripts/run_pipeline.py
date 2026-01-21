#!/usr/bin/env python3
"""Full 3-Stage Pipeline: Preference Extraction → SFT → DPO"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from data.movielens import MovieLensDataset
from src.preference_extraction import PreferenceExtractor
from src.preference_pairs import PreferencePairConstructor
from src.sft_trainer import SFTTrainer, SFTConfig
from src.dpo_trainer import DPORecommendationTrainer, DPOTrainerConfig
from src.evaluation import Evaluator


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_pipeline(config_path: str, max_samples: int = None, skip_sft: bool = False, skip_dpo: bool = False):
    print("=" * 60)
    print("Debiased LLM Recommendation Pipeline")
    print("=" * 60)
    
    config = load_config(config_path)
    print(f"\nConfig: {config_path}")
    print(f"Model: {config.get('model_name')}")
    
    # Stage 1: Data Preparation
    print("\n[Stage 1] Data Preparation")
    dataset = MovieLensDataset()
    stats = dataset.get_stats()
    print(f"Users: {stats['num_users']}, Items: {stats['num_items']}, Ratings: {stats['num_ratings']}")
    
    extractor = PreferenceExtractor(dataset)
    constructor = PreferencePairConstructor(dataset)
    user_ids = dataset.get_all_users()
    if max_samples:
        user_ids = user_ids[:max_samples]
    
    print(f"Generating data for {len(user_ids)} users...")
    sft_data = constructor.generate_sft_dataset(extractor, user_ids)
    dpo_data = constructor.generate_dpo_dataset(extractor, user_ids)
    print(f"SFT samples: {len(sft_data)}, DPO pairs: {len(dpo_data)}")
    
    split = int(len(sft_data) * 0.9)
    sft_train, sft_val = sft_data[:split], sft_data[split:]
    dpo_train, dpo_val = dpo_data[:split], dpo_data[split:]
    
    sft_dir = config.get('sft', {}).get('output_dir', './outputs/sft')
    dpo_dir = config.get('dpo', {}).get('output_dir', './outputs/dpo')
    
    # Stage 2: SFT
    if not skip_sft:
        print("\n[Stage 2] Supervised Fine-Tuning")
        sft_cfg = SFTConfig(
            model_name=config.get('model_name'), max_seq_length=config.get('max_seq_length', 256),
            lora_r=config.get('lora_r', 16), lora_alpha=config.get('lora_alpha', 32),
            lora_dropout=config.get('lora_dropout', 0.05), target_modules=config.get('target_modules', ['q_proj', 'v_proj']),
            **config.get('sft', {})
        )
        SFTTrainer(sft_cfg).train(sft_train, sft_val)
    
    # Stage 3: DPO
    if not skip_dpo:
        print("\n[Stage 3] Direct Preference Optimization")
        dpo_cfg = DPOTrainerConfig(
            model_name=config.get('model_name'), sft_model_path=sft_dir, max_seq_length=config.get('max_seq_length', 256),
            lora_r=config.get('lora_r', 16), lora_alpha=config.get('lora_alpha', 32),
            lora_dropout=config.get('lora_dropout', 0.05), target_modules=config.get('target_modules', ['q_proj', 'v_proj']),
            **config.get('dpo', {})
        )
        DPORecommendationTrainer(dpo_cfg).train(dpo_train, dpo_val)
    
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print(f"SFT: {sft_dir}, DPO: {dpo_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run debiased LLM recommendation pipeline")
    parser.add_argument("--config", "-c", default="configs/test_tiny.yaml")
    parser.add_argument("--max-samples", "-n", type=int, default=None)
    parser.add_argument("--skip-sft", action="store_true")
    parser.add_argument("--skip-dpo", action="store_true")
    args = parser.parse_args()
    run_pipeline(args.config, args.max_samples, args.skip_sft, args.skip_dpo)


if __name__ == "__main__":
    main()
