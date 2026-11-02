"""The model-under-training. Samples k candidates per decision point.

Keep inference-time prompt construction identical to data/build_pairs._render_prompt,
or you create train/serve skew.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass


@dataclass
class GenConfig:
    model_path: str = "./registry/policy-current"
    k: int = 4                 # candidates per step — the preference set width
    temperature: float = 0.9   # need diversity across the k, not the single best
    top_p: float = 0.95
    max_new_tokens: int = 512


class Policy:
    def __init__(self, cfg: GenConfig):
        self.cfg = cfg
        self._model = None
        self._tok = None

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(self.cfg.model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_path, device_map="auto", torch_dtype="auto",
        )

    def sample(self, system_prompt: str, prompt: str) -> list[dict]:
        """Return k candidates: [{cand_id, text, logprob}]."""
        if self._model is None:
            self.load()
        import torch

        msgs = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}]
        inputs = self._tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt",
        ).to(self._model.device)

        out = self._model.generate(
            inputs,
            do_sample=True,
            num_return_sequences=self.cfg.k,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_new_tokens=self.cfg.max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
        )
        gen = out.sequences[:, inputs.shape[1]:]
        texts = self._tok.batch_decode(gen, skip_special_tokens=True)

        cands = []
        for t in texts:
            cands.append({"cand_id": str(uuid.uuid4())[:8], "text": t.strip(),
                          "logprob": None})
        return cands
