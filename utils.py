import os, re, json
import torch, numpy
from collections import defaultdict
from util import nethook
from util.globals import DATA_DIR
from experiments.causal_trace import (
    ModelAndTokenizer,
    layername,
    guess_subject,
    plot_trace_heatmap, calculate_hidden_flow, plot_hidden_flow,
)
from experiments.causal_trace import (
    make_inputs,
    decode_tokens,
    find_token_range,
    predict_token,
    predict_from_input,
    collect_embedding_std,
)
from dsets import KnownsDataset

torch.set_grad_enabled(False)

class RomeTest:
    def __init__(self, model_name, noise_level=None):
        self.mt = ModelAndTokenizer(
            model_name,
            torch_dtype=(torch.float16 if "20b" in model_name else None),
        )
        self.mt.tokenizer.pad_token = self.mt.tokenizer.eos_token

        if noise_level:
            self.noise_level = noise_level
        else:
            knowns = KnownsDataset(DATA_DIR)  # Dataset of known facts
            self.noise_level = 3 * collect_embedding_std(self.mt, [k["subject"] for k in knowns])
            print(f"noise level: {self.noise_level}")

    def get_max_impact_layer(self, prompt, kind="mlp"):
        subject = guess_subject(prompt)
        results = calculate_hidden_flow(self.mt, prompt, subject, kind=kind, noise=self.noise_level)
        max_effect_layer = int((results["scores"]==torch.max(results["scores"])).nonzero()[0][1] + 1)
        return max_effect_layer

    def plot_causal_tracing_effect(self, prompt, kind="mlp"):
        plot_hidden_flow(self.mt, prompt, kind=kind)

if __name__ == '__main__':
    rome_test = RomeTest("google/gemma-2-2b", noise_level=0.15380)
    rome_test.plot_causal_tracing_effect("Steve Jobs was the founder of", kind="mlp")
