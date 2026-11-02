"""SFT on demonstration transcripts. Run this FIRST, before any DPO.

Input: JSONL of {"messages": [...]} chat transcripts of the pentester working
with (or demonstrating for) the agent. DPO assumes a policy that already
produces plausible candidates; cold-starting DPO on a base model wastes the
preference data.
"""
from __future__ import annotations

import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def _optional_qlora(cfg):
    """Return (from_pretrained kwargs, peft_config). Both empty/None if no LoRA."""
    if "lora" not in cfg:
        return {}, None
    from peft import LoraConfig
    quant = {}
    if cfg.get("load_in_4bit"):
        from transformers import BitsAndBytesConfig
        import torch
        quant["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    l = cfg["lora"]
    peft_cfg = LoraConfig(
        r=l["r"], lora_alpha=l["alpha"], lora_dropout=l.get("dropout", 0.05),
        target_modules=l["target_modules"], task_type="CAUSAL_LM", bias="none",
    )
    return quant, peft_cfg


def main(cfg_path: str = "config/sft.yaml"):
    cfg = yaml.safe_load(open(cfg_path))

    model_id = cfg["base_model"]
    quant_args, peft_cfg = _optional_qlora(cfg)
    if cfg.get("load_in_4bit") is False and "base_model_small" in cfg:
        model_id = cfg["base_model_small"]   # ROCm bf16-LoRA fallback path

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto", **quant_args,
    )

    ds = load_dataset("json", data_files=cfg["train_file"], split="train")

    args = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg.get("epochs", 2),
        per_device_train_batch_size=cfg.get("batch_size", 4),
        gradient_accumulation_steps=cfg.get("grad_accum", 8),
        learning_rate=cfg.get("lr", 1e-5),
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_seq_length=cfg.get("max_seq_len", 4096),
        packing=cfg.get("packing", True),
    )

    trainer = SFTTrainer(model=model, args=args, train_dataset=ds,
                         processing_class=tok, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tok.save_pretrained(cfg["output_dir"])
    _record_provenance(cfg, model_id)
    print(f"SFT done -> {cfg['output_dir']}")


def _record_provenance(cfg, base_model):
    """Write a registry manifest so we can trace which data shaped this model."""
    try:
        import os
        from rlhf import registry
        version = os.path.basename(cfg["output_dir"].rstrip("/"))
        registry.record(version, stage="sft", base_model=base_model,
                        data_file=cfg["train_file"], config=cfg)
        print(f"recorded provenance: registry/{version}/manifest.json")
    except Exception as e:  # provenance is best-effort; never fail a finished run
        print(f"warning: could not record provenance: {e}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "config/sft.yaml")
