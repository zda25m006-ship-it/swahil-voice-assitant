"""
Evaluate ASR WER/CER for your own small Swahili tourist audio test set.

Create a CSV like:
    audio_path,reference
    eval_audio/q1.wav,Nataka kutembelea Stone Town
    eval_audio/q2.wav,Ni wapi naweza kuona kima punju

Run:
    pip install -r requirements-train.txt
    python scripts/evaluate_asr.py --manifest data/asr_eval_manifest.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from jiwer import cer, wer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import transcribe_audio  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="CSV with audio_path,reference")
    parser.add_argument("--out", default="report/asr_eval_results.csv")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    rows = []
    refs, hyps = [], []
    for _, row in manifest.iterrows():
        audio_path = str(row["audio_path"])
        if not Path(audio_path).is_absolute():
            audio_path = str(ROOT / audio_path)
        ref = str(row["reference"])
        hyp = transcribe_audio(audio_path)
        refs.append(ref)
        hyps.append(hyp)
        rows.append({"audio_path": row["audio_path"], "reference": ref, "prediction": hyp, "wer": wer(ref, hyp), "cer": cer(ref, hyp)})
        print(f"REF: {ref}\nHYP: {hyp}\n")

    out = pd.DataFrame(rows)
    out_path = ROOT / args.out
    out_path.parent.mkdir(exist_ok=True)
    out.to_csv(out_path, index=False)
    print("ASR Evaluation")
    print("==============")
    print(f"Files: {len(rows)}")
    print(f"Corpus WER: {wer(refs, hyps):.3f}")
    print(f"Corpus CER: {cer(refs, hyps):.3f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
