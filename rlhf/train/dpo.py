"""DPO on the preference pairs from data/build_pairs.py.

Starts from the SFT checkpoint, not the base model. The `weight` column (signal
strength x outcome confidence) is applied via per-example loss weighting; if you
want to skip that complexity initially, just drop weights and train uniform —
it still works, you just lose the "edited-and-it-worked" examples counting for
more than "rated 4/5".
"""
from __future__ import annotations

import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def _optional_qlora(cfg):
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


def main(cfg_path: str = "config/dpo.yaml"):
    cfg = yaml.safe_load(open(cfg_path))

    quant_args, peft_cfg = _optional_qlora(cfg)
    tok = AutoTokenizer.from_pretrained(cfg["sft_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["sft_model"], torch_dtype="auto", device_map="auto", **quant_args,
    )

    # DPOTrainer wants columns: prompt, chosen, rejected. We keep weight/meta
    # around for optional loss weighting / auditing.
    ds = load_dataset("json", data_files=cfg["pairs_file"], split="train")

    args = DPOConfig(
        output_dir=cfg["output_dir"],
        beta=cfg.get("beta", 0.1),              # KL strength; 0.1 is a sane start
        num_train_epochs=cfg.get("epochs", 1),  # DPO overfits fast — start at 1
        per_device_train_batch_size=cfg.get("batch_size", 2),
        gradient_accumulation_steps=cfg.get("grad_accum", 16),
        learning_rate=cfg.get("lr", 5e-7),      # much smaller than SFT
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_length=cfg.get("max_len", 4096),
        max_prompt_length=cfg.get("max_prompt_len", 3072),
        precompute_ref_log_probs=cfg.get("precompute_ref_log_probs", False),
    )

    # With LoRA, ref_model=None makes TRL use the base (adapters disabled) as the
    # implicit reference — the only way DPO fits alongside the policy on 16GB.
    trainer = DPOTrainer(model=model, ref_model=None, args=args,
                         train_dataset=ds, processing_class=tok, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tok.save_pretrained(cfg["output_dir"])
    _record_provenance(cfg)
    print(f"DPO done -> {cfg['output_dir']}")


def _record_provenance(cfg):
    """Manifest with the SFT parent + the exact pair file (by hash) this used."""
    try:
        import os
        from rlhf import registry
        version = os.path.basename(cfg["output_dir"].rstrip("/"))
        registry.record(version, stage="dpo",
                        parent=os.path.basename(str(cfg.get("sft_model", "")).rstrip("/")),
                        data_file=cfg["pairs_file"], config=cfg)
        print(f"recorded provenance: registry/{version}/manifest.json")
    except Exception as e:
        print(f"warning: could not record provenance: {e}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "config/dpo.yaml")
