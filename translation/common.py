"""Shared helpers for the LLM-JEPA low-resource translation experiment."""
import json
import random

import torch

LANG_NAMES = {"asm_Beng": "Assamese", "brx_Deva": "Bodo", "mni_Mtei": "Manipuri"}


def pick_device(name=""):
    if name:
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def prompt_text(src, lang_name):
    """The translation prompt (also the first JEPA 'view')."""
    return f"Translate English to {lang_name}.\nEnglish: {src}\n{lang_name}:"


def load_model(model_name, device, dtype_name="float32"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"torch_dtype": dtype}
    if "gemma-2" in model_name.lower():
        kw["attn_implementation"] = "eager"  # recommended for Gemma-2
    model = AutoModelForCausalLM.from_pretrained(model_name, **kw).to(device)
    return tok, model
