"""
Offline evaluation harness for the Video-QA pipeline.

Metrics reported:
  - Retrieval precision@5    (answered queries: expected_keywords hit in top-5 chunks)
  - Answer accuracy          (answered queries: expected_keywords hit in the answer)
  - Hallucination rate       (answered queries whose post-check returns UNSUPPORTED)
  - Refusal correctness      (out-of-scope queries correctly refused)
  - Mean confidence / mean support score
  - Cache hit rate           (run dataset twice)

Usage:
    python -m evaluation.run_eval
    python -m evaluation.run_eval --dataset evaluation/eval_dataset.json --json out.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Make project root importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _quiet_logging() -> None:
    logging.disable(logging.WARNING)
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


def _kw_hit(answer: str, keywords: List[str]) -> bool:
    if not keywords:
        return True
    a = (answer or "").lower()
    return all(k.lower() in a for k in keywords)


def _kw_in_any_chunk(chunks: List[Dict[str, Any]], keywords: List[str]) -> bool:
    if not keywords:
        return True
    blob = " ".join((c.get("text") or "") for c in chunks).lower()
    return all(k.lower() in blob for k in keywords)


def _is_refusal(answer: str) -> bool:
    a = (answer or "").lower()
    return ("not found in video" in a) or ("cannot be answered" in a)


def run(
    dataset_path: str = "evaluation/eval_dataset.json",
    json_out: str = None,
) -> Dict[str, Any]:
    _quiet_logging()

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    queries = data["queries"]

    from video_qa.pipeline import VideoQAPipeline
    from video_qa.cache import get_cache

    cache = get_cache()
    cache.clear()

    pipeline = VideoQAPipeline()

    per_query: List[Dict[str, Any]] = []

    for q in queries:
        print(f"[eval] ({q['id']}) asking: {q['query'][:60]}", flush=True)
        t0 = time.time()
        result = pipeline.ask(q["query"], use_cache=True, include_neighbors=True)
        dt = round(time.time() - t0, 3)
        print(f"[eval] ({q['id']}) done in {dt}s status={result.get('status')} conf={result.get('confidence')} provider={result.get('provider')}", flush=True)

        answer = result.get("answer", "")
        contexts = result.get("contexts", []) or []
        support = result.get("support", {}) or {}

        record = {
            "id":                q["id"],
            "query":             q["query"],
            "out_of_scope":      q.get("out_of_scope", False),
            "expected_keywords": q.get("expected_keywords", []),
            "answer":            answer,
            "provider":          result.get("provider"),
            "confidence":        result.get("confidence"),
            "status":            result.get("status"),
            "timestamp":         result.get("timestamp"),
            "chunk_ids":         result.get("chunk_ids"),
            "semantic_sim":      support.get("semantic_similarity"),
            "keyword_overlap":   support.get("keyword_overlap"),
            "support_score":     support.get("support_score"),
            "latency_s":         dt,
            "refused":           _is_refusal(answer),
        }

        # Metric flags
        if q.get("out_of_scope"):
            # A correct refusal = said "Not found" or status UNSUPPORTED
            record["refusal_correct"] = record["refused"] or record["status"] == "UNSUPPORTED"
            record["answer_accurate"] = None
            record["retrieval_precision"] = None
        else:
            record["refusal_correct"] = None
            record["answer_accurate"]  = _kw_hit(answer, q.get("expected_keywords", []))
            record["retrieval_precision"] = _kw_in_any_chunk(contexts, q.get("expected_keywords", []))

        per_query.append(record)

    # ── Second pass to measure cache hit rate ─────────────────────────────
    stats_before = cache.stats()
    t_cache_start = time.time()
    for q in queries:
        pipeline.ask(q["query"], use_cache=True)
    t_cache_total = round(time.time() - t_cache_start, 3)
    stats_after = cache.stats()

    # ── Aggregate metrics ─────────────────────────────────────────────────
    in_scope   = [r for r in per_query if not r["out_of_scope"]]
    out_scope  = [r for r in per_query if r["out_of_scope"]]

    def _rate(lst: List[bool]) -> float:
        if not lst:
            return 0.0
        return round(sum(1 for x in lst if x) / len(lst), 4)

    metrics = {
        "n_total":                len(per_query),
        "n_in_scope":             len(in_scope),
        "n_out_of_scope":         len(out_scope),
        "retrieval_precision":    _rate([r["retrieval_precision"] for r in in_scope]),
        "answer_accuracy":        _rate([r["answer_accurate"]       for r in in_scope]),
        "hallucination_rate":     _rate([r["status"] == "UNSUPPORTED" for r in in_scope]),
        "supported_rate":         _rate([r["status"] == "SUPPORTED"   for r in in_scope]),
        "refusal_correctness":    _rate([r["refusal_correct"] for r in out_scope]) if out_scope else None,
        "mean_confidence":        round(
            sum((r["confidence"] or 0) for r in in_scope) / max(1, len(in_scope)), 2
        ),
        "mean_support_score":     round(
            sum((r["support_score"] or 0) for r in in_scope) / max(1, len(in_scope)), 4
        ),
        "mean_latency_s":         round(
            sum(r["latency_s"] for r in per_query) / max(1, len(per_query)), 3
        ),
        "cache_hits_second_pass": stats_after["hits"] - stats_before["hits"],
        "second_pass_latency_s":  t_cache_total,
    }

    report = {"metrics": metrics, "per_query": per_query}

    print("\n" + "=" * 70)
    print("VIDEO-QA EVALUATION REPORT")
    print("=" * 70)
    for k, v in metrics.items():
        print(f"  {k:28s}: {v}")
    print("=" * 70)
    print(f"{'id':8s} {'status':12s} {'conf':5s} {'acc':4s} {'ret':4s} query")
    print("-" * 70)
    for r in per_query:
        acc = "-" if r["answer_accurate"] is None else ("✓" if r["answer_accurate"] else "✗")
        ret = "-" if r["retrieval_precision"] is None else ("✓" if r["retrieval_precision"] else "✗")
        print(f"{r['id']:8s} {str(r['status']):12s} {str(r['confidence']):5s} {acc:4s} {ret:4s} {r['query'][:40]}")
    print("=" * 70)

    if json_out:
        Path(json_out).write_text(json.dumps(report, indent=2))
        print(f"[saved] {json_out}")

    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="evaluation/eval_dataset.json")
    p.add_argument("--json", dest="json_out", default=None)
    args = p.parse_args()
    run(args.dataset, args.json_out)


if __name__ == "__main__":
    main()
