"""Stage 2: Supervised Fine-Tuning with LoRA."""

import torch
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path

from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType


@dataclass
class SFTConfig:
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    max_seq_length: int = 512
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    learning_rate: float = 2e-5
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    output_dir: str = "./outputs/sft"
    logging_steps: int = 10
    save_steps: int = 100
    device: str = "auto"
    
    @classmethod
    def from_yaml(cls, path: str) -> "SFTConfig":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get('sft', data))


class SFTDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = f"### Instruction:\n{item['prompt']}\n\n### Response:\n{item['completion']}"
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"].squeeze()
        attn_mask = enc["attention_mask"].squeeze()
        labels = input_ids.clone()
        labels[attn_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}


class SFTTrainer:
    def __init__(self, config: Union[SFTConfig, dict, str]):
        if isinstance(config, str):
            config = SFTConfig.from_yaml(config)
        elif isinstance(config, dict):
            config = SFTConfig(**config)
        self.config = config
        self.model = None
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
    
    def load_model(self):
        print(f"Loading model: {self.config.model_name}")
        print(f"Device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        dtype = torch.float16 if self.device in ["cuda", "mps"] else torch.float32
        kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
        if self.device == "cuda":
            kwargs["device_map"] = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name, **kwargs)
        
        lora_cfg = LoraConfig(
            r=self.config.lora_r, lora_alpha=self.config.lora_alpha, lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules, task_type=TaskType.CAUSAL_LM, bias="none"
        )
        self.model = get_peft_model(self.model, lora_cfg)
        
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        return self.model
    
    def train(self, train_data: List[Dict], val_data: Optional[List[Dict]] = None, max_samples: Optional[int] = None):
        if self.model is None:
            self.load_model()
        
        if max_samples:
            train_data = train_data[:max_samples]
            if val_data:
                val_data = val_data[:max_samples // 5]
        
        print(f"Training on {len(train_data)} samples")
        train_ds = SFTDataset(train_data, self.tokenizer, self.config.max_seq_length)
        val_ds = SFTDataset(val_data, self.tokenizer, self.config.max_seq_length) if val_data else None
        
        args = TrainingArguments(
            output_dir=self.config.output_dir, num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate, warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay, logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps, save_total_limit=2,
            fp16=(self.device == "cuda"), report_to="none", remove_unused_columns=False
        )
        
        collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        trainer = Trainer(model=self.model, args=args, train_dataset=train_ds, eval_dataset=val_ds, data_collator=collator)
        
        print("Starting training...")
        trainer.train()
        print(f"Saving model to {self.config.output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        return trainer
    
    def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.7) -> str:
        if self.model is None:
            self.load_model()
        
        full = f"### Instruction:\n{prompt}\n\n### Response:\n"
        inputs = self.tokenizer(full, return_tensors="pt", truncation=True, max_length=self.config.max_seq_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)
        
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        if "### Response:" in text:
            text = text.split("### Response:")[-1].strip()
        return text
    
    @classmethod
    def load_trained(cls, model_path: str, config: Optional[SFTConfig] = None):
        from peft import PeftModel
        if config is None:
            config = SFTConfig()
        trainer = cls(config)
        trainer.tokenizer = AutoTokenizer.from_pretrained(model_path)
        dtype = torch.float16 if trainer.device in ["cuda", "mps"] else torch.float32
        base = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=dtype, trust_remote_code=True)
        trainer.model = PeftModel.from_pretrained(base, model_path).to(trainer.device)
        return trainer


if __name__ == "__main__":
    cfg = SFTConfig(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", max_seq_length=256, num_epochs=1, batch_size=2)
    trainer = SFTTrainer(cfg)
    print(f"Device: {trainer.device}")
