"""Stage 3: Direct Preference Optimization with TRL."""

import torch
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field

from datasets import Dataset as HFDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from trl import DPOTrainer as TRLDPOTrainer, DPOConfig


@dataclass
class DPOTrainerConfig:
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    sft_model_path: Optional[str] = None
    max_seq_length: int = 512
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    beta: float = 0.1
    learning_rate: float = 5e-6
    num_epochs: int = 1
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    output_dir: str = "./outputs/dpo"
    logging_steps: int = 10
    save_steps: int = 100
    device: str = "auto"
    
    @classmethod
    def from_yaml(cls, path: str) -> "DPOTrainerConfig":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get('dpo', data))


class DPORecommendationTrainer:
    """DPO trainer for debiasing recommendations toward low-popularity items."""
    
    def __init__(self, config: Union[DPOTrainerConfig, dict, str]):
        if isinstance(config, str):
            config = DPOTrainerConfig.from_yaml(config)
        elif isinstance(config, dict):
            config = DPOTrainerConfig(**config)
        self.config = config
        self.model = None
        self.ref_model = None
        self.tokenizer = None
        self.device = self._get_device()
    
    def _get_device(self) -> str:
        if self.config.device != "auto":
            return self.config.device
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def load_models(self):
        print(f"Loading models for DPO training")
        print(f"Device: {self.device}")
        
        tokenizer_path = self.config.sft_model_path or self.config.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        dtype = torch.float16 if self.device in ["cuda", "mps"] else torch.float32
        kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
        
        print(f"Loading base model: {self.config.model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(self.config.model_name, **kwargs)
        
        if self.config.sft_model_path:
            print(f"Loading SFT weights from: {self.config.sft_model_path}")
            self.model = PeftModel.from_pretrained(base_model, self.config.sft_model_path)
            self.model = self.model.merge_and_unload()
        else:
            self.model = base_model
        
        print("Loading reference model...")
        ref_base = AutoModelForCausalLM.from_pretrained(self.config.model_name, **kwargs)
        if self.config.sft_model_path:
            self.ref_model = PeftModel.from_pretrained(ref_base, self.config.sft_model_path)
            self.ref_model = self.ref_model.merge_and_unload()
        else:
            self.ref_model = ref_base
        
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        lora_cfg = LoraConfig(
            r=self.config.lora_r, lora_alpha=self.config.lora_alpha, lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules, task_type=TaskType.CAUSAL_LM, bias="none"
        )
        self.model = get_peft_model(self.model, lora_cfg)
        
        if self.device != "cuda":
            self.model = self.model.to(self.device)
            self.ref_model = self.ref_model.to(self.device)
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        return self.model
    
    def _prepare_dataset(self, data: List[Dict]) -> HFDataset:
        formatted = [{'prompt': f"### Instruction:\n{d['prompt']}\n\n### Response:\n", 
                      'chosen': d['chosen'], 'rejected': d['rejected']} for d in data]
        return HFDataset.from_list(formatted)
    
    def train(self, train_data: List[Dict], val_data: Optional[List[Dict]] = None, max_samples: Optional[int] = None):
        if self.model is None:
            self.load_models()
        
        if max_samples:
            train_data = train_data[:max_samples]
            if val_data:
                val_data = val_data[:max_samples // 5]
        
        print(f"Training DPO on {len(train_data)} pairs")
        train_ds = self._prepare_dataset(train_data)
        eval_ds = self._prepare_dataset(val_data) if val_data else None
        
        args = DPOConfig(
            output_dir=self.config.output_dir, num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate, warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay, logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps, save_total_limit=2,
            fp16=(self.device == "cuda"), report_to="none", remove_unused_columns=False,
            beta=self.config.beta, max_length=self.config.max_seq_length, max_prompt_length=self.config.max_seq_length // 2
        )
        
        trainer = TRLDPOTrainer(
            model=self.model, ref_model=self.ref_model, args=args,
            train_dataset=train_ds, eval_dataset=eval_ds, processing_class=self.tokenizer
        )
        
        print("Starting DPO training...")
        trainer.train()
        print(f"Saving model to {self.config.output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        return trainer
    
    def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.7) -> str:
        if self.model is None:
            self.load_models()
        
        full = f"### Instruction:\n{prompt}\n\n### Response:\n"
        inputs = self.tokenizer(full, return_tensors="pt", truncation=True, max_length=self.config.max_seq_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)
        
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        if "### Response:" in text:
            text = text.split("### Response:")[-1].strip()
        return text


if __name__ == "__main__":
    cfg = DPOTrainerConfig(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", max_seq_length=256, num_epochs=1, batch_size=1, beta=0.1)
    trainer = DPORecommendationTrainer(cfg)
    print(f"Device: {trainer.device}")
