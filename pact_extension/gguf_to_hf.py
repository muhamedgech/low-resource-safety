"""Convert a GGUF checkpoint to a standard Hugging Face folder (dequantized to bf16).

The repo's loader (nnsight) needs a normal HF directory with config.json + safetensors.
A .gguf file (llama.cpp format) is not that, so we load it through transformers'
GGUF support once and re-save it as a plain HF checkpoint that every script can use.

Usage (offline, on the server):
    uv pip install gguf                       # transformers needs this to read GGUF
    uv run python pact_extension/gguf_to_hf.py \
        --gguf /data/BL25215003/pytorch_lab/models/Llama-3.1-8B-Instruct-8bit/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf \
        --out  /data/BL25215003/pytorch_lab/models/Llama-3.1-8B-Instruct-hf

Then extract activations with  model.name=<the --out folder>.
The output is ~16 GB (bf16), so make sure there is room on the data disk.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gguf", required=True, help="path to the .gguf file")
    p.add_argument("--out", required=True, help="output HF folder to create")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gguf_path = Path(args.gguf)
    gguf_dir, gguf_file = str(gguf_path.parent), gguf_path.name
    dtype = getattr(torch, args.dtype)

    print(f"loading {gguf_file} (dequantizing to {args.dtype}) ...")
    # transformers reads the GGUF via the directory + gguf_file pair.
    model = AutoModelForCausalLM.from_pretrained(gguf_dir, gguf_file=gguf_file, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(gguf_dir, gguf_file=gguf_file)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"saving HF checkpoint to {out} ...")
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    has_type = (out / "config.json").read_text().find('"model_type"') >= 0
    print(f"done. config.json has model_type: {has_type}")
    print(f"use it with:  model.name={out}")


if __name__ == "__main__":
    main()
