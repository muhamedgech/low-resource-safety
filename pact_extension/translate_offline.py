"""Offline PolyRefuse translation with NLLB — a drop-in replacement for the repo's
googletrans-based `scripts/setup/polyrefuse_translate.py`.

Why: the original translator calls Google Translate, which is unreachable from some
networks (e.g. behind the GFW). NLLB-200 runs locally, so this works offline once the
model is cached. It writes byte-identical file schema, so the authors'
`extract_activations.py` and everything downstream consume the output unchanged.

Output files:  {root}/{subset}_{split}_translated_{lang}.json
Each row:      {instruction (target text), category, source_id, source_instruction (English)}

Usage (on the GPU/CPU server, after polyrefuse_download.py):
    # cache the model once (China: set HF_ENDPOINT=https://hf-mirror.com first)
    python pact_extension/translate_offline.py --target-languages am sw my km si
    # then extract activations as usual, e.g. for Amharic:
    #   python scripts/hrl_direction/extract_activations.py \
    #       model=llama-3.1-8b-instruct 'dataset.languages=[en,am]' 'extraction.layers=[10]'
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# code used in PolyRefuse file names  ->  NLLB-200 language code
NLLB_CODE = {
    "en": "eng_Latn",
    "am": "amh_Ethi",   # Amharic
    "sw": "swh_Latn",   # Swahili
    "my": "mya_Mymr",   # Burmese
    "km": "khm_Khmr",   # Khmer
    "si": "sin_Sinh",   # Sinhala
    "yo": "yor_Latn",   # Yoruba
}


class NLLB:
    """Loads NLLB once and translates batches of strings between two languages."""

    def __init__(self, model_name: str, device: str | None = None):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device).eval()

    def translate(self, texts: list[str], src: str, tgt: str, batch_size: int = 16,
                  max_new_tokens: int = 128) -> list[str]:
        import torch

        self.tokenizer.src_lang = src
        bos = self.tokenizer.convert_tokens_to_ids(tgt)
        out: list[str] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                gen = self.model.generate(**enc, forced_bos_token_id=bos,
                                          max_new_tokens=max_new_tokens, num_beams=1, do_sample=False)
            out.extend(self.tokenizer.batch_decode(gen, skip_special_tokens=True))
            print(f"    {min(i + batch_size, len(texts))}/{len(texts)}", end="\r")
        print()
        return [t.strip() for t in out]


def in_script_fraction(text: str, lang: str) -> float:
    """Fraction of letters in the target script — a quick 'did it really translate?' check."""
    ranges = {"am": [(0x1200, 0x137F)], "my": [(0x1000, 0x109F)], "km": [(0x1780, 0x17FF)],
              "si": [(0x0D80, 0x0DFF)]}.get(lang)
    letters = [c for c in text if c.isalpha()]
    if not ranges or not letters:
        return 1.0  # Latin-script targets (sw, yo) skip this check
    return sum(any(lo <= ord(c) <= hi for lo, hi in ranges) for c in letters) / len(letters)


def translate_language(root: Path, lang: str, translator: NLLB, subsets, splits, overwrite: bool):
    tgt = NLLB_CODE[lang]
    for subset in subsets:
        for split in splits:
            src_path = root / f"{subset}_{split}_translated_en.json"
            dst_path = root / f"{subset}_{split}_translated_{lang}.json"
            if not src_path.exists():
                print(f"[miss] {src_path.name} — skip")
                continue
            if dst_path.exists() and not overwrite:
                print(f"[skip] {dst_path.name} exists (use --overwrite)")
                continue
            items = [r for r in json.loads(src_path.read_text()) if r.get("instruction")]
            print(f"[run ] {src_path.name} -> {dst_path.name}  ({len(items)} items)")
            english = [str(r["instruction"]) for r in items]
            translated = translator.translate(english, NLLB_CODE["en"], tgt)
            good = sum(in_script_fraction(t, lang) >= 0.5 for t in translated)
            print(f"       in-{lang}-script: {good}/{len(translated)}")
            rows = [{"instruction": t, "category": r.get("category"),
                     "source_id": r.get("source_id", str(i)),
                     "source_instruction": r.get("source_instruction", r.get("instruction"))}
                    for i, (r, t) in enumerate(zip(items, translated))]
            dst_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("data/polyrefuse"))
    p.add_argument("--target-languages", nargs="+", required=True, choices=[c for c in NLLB_CODE if c != "en"])
    p.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    p.add_argument("--subsets", nargs="+", default=["harmful", "harmless"])
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    print(f"loading {args.model} ...")
    translator = NLLB(args.model)
    for lang in args.target_languages:
        translate_language(args.root, lang, translator, args.subsets, args.splits, args.overwrite)
    print("done")


if __name__ == "__main__":
    main()
