import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.2-1B"

model, tok = (
    AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(
        "cuda"
    ),
    AutoTokenizer.from_pretrained(MODEL_NAME),
)
tok.pad_token = tok.eos_token

from rome import ROMEHyperParams, apply_rome_to_model

request = [
    {
        "prompt": "{} was the founder of",
        "subject": "Steve Jobs",
        "target_new": {"str": "Microsoft"},
    }
]

torch.set_grad_enabled(True)

hparams = ROMEHyperParams.from_json("hparams/ROME/meta-llama_llama3-2-1b.json")
edited_model, weights_copy = apply_rome_to_model(model, tok, request, hparams)