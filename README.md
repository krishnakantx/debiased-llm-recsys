# Debiased LLM Recommendation System

Implementation of *Multi-Stage Alignment of Large Language Models for Popularity Bias Mitigation in Generative Movie Recommendation*.

## Quick Start

```bash
pip install -r requirements.txt

# Local test (TinyLlama, 50 samples)
python scripts/run_pipeline.py --config configs/test_tiny.yaml --max-samples 50

# Production (GPU)
python scripts/run_pipeline.py --config configs/mistral_7b.yaml
```

## Project Structure

```
data/           Dataset loaders (MovieLens 1M)
src/            Core modules (preference extraction, SFT, DPO, evaluation)
configs/        YAML configurations (TinyLlama, Mistral 7B, LLaMA)
scripts/        Executable scripts (run_pipeline, run_sft, run_dpo, run_eval)
```

## Pipeline

1. **Preference Extraction** — User histories → prompts
2. **SFT** — Behavioral grounding with LoRA
3. **DPO** — Bias correction toward low-popularity items

## Evaluation Metrics

| Quality | Bias |
|---------|------|
| Precision@K | Novelty |
| nDCG@K | Coverage |
| MAP@K | Avg Popularity |

## License

MIT
