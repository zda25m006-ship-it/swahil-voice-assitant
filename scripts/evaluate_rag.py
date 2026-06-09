"""
Evaluate retrieval Top-1/Top-3 accuracy on data/evaluation_questions.csv.

Run:
    python scripts/evaluate_rag.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import get_smalltalk_answer, detect_lang, retrieve  # noqa: E402


def main():
    eval_path = ROOT / "data" / "evaluation_questions.csv"
    df = pd.read_csv(eval_path).fillna("")
    rows = []
    top1_correct = 0
    top3_correct = 0
    evaluated = 0

    for _, row in df.iterrows():
        q = row["question"]
        expected = row["expected_place"]
        expected_category = row["expected_category"]
        lang = row["language"] or detect_lang(q)

        if expected_category == "smalltalk":
            ans = get_smalltalk_answer(q, lang)
            ok = bool(ans)
            rows.append({"question": q, "expected": "smalltalk", "top1": "SMALLTALK", "top3": "SMALLTALK", "ok": ok})
            top1_correct += int(ok)
            top3_correct += int(ok)
            evaluated += 1
            continue

        results = retrieve(q, top_k=3)
        names = [r[0]["name"] for r in results]
        ok1 = len(names) > 0 and names[0].lower() == expected.lower()
        ok3 = expected.lower() in [n.lower() for n in names]
        rows.append({"question": q, "expected": expected, "top1": names[0] if names else "", "top3": " | ".join(names), "ok": ok3})
        top1_correct += int(ok1)
        top3_correct += int(ok3)
        evaluated += 1

    out = pd.DataFrame(rows)
    out_path = ROOT / "report" / "rag_eval_results.csv"
    out_path.parent.mkdir(exist_ok=True)
    out.to_csv(out_path, index=False)

    print("RAG Evaluation")
    print("==============")
    print(f"Questions: {evaluated}")
    print(f"Top-1 accuracy: {top1_correct / evaluated:.3f}")
    print(f"Top-3 accuracy: {top3_correct / evaluated:.3f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
