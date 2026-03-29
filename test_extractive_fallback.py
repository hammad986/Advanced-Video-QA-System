#!/usr/bin/env python3
"""
Integration test for the extractive fallback fix.

Verifies that when ALL LLM providers fail but valid context chunks exist,
the system returns a grounded answer from the top context chunk
instead of "NOT FOUND IN VIDEO".

This test does NOT require:
  - Ollama running
  - Any API keys
  - Network access

It patches all LLM callers to return None, simulating a total LLM failure.
"""

import sys
import unittest
from unittest.mock import patch, MagicMock

# Ensure the video_qa package is importable
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))


def test_extractive_fallback_direct():
    """Test _extractive_fallback() in isolation."""
    print("\n" + "="*70)
    print("TEST 1: _extractive_fallback() — direct unit test")
    print("="*70)

    from video_qa.answer_generator import _extractive_fallback

    contexts = [
        {
            "text": "Supervised learning is a type of machine learning where the model is trained on labeled data. The goal is to learn a mapping from inputs to outputs.",
            "start": 45.0,
            "end": 62.0,
            "score": 0.92,
        },
        {
            "text": "Regularization is a technique to prevent overfitting.",
            "start": 70.0,
            "end": 80.0,
            "score": 0.80,
        },
    ]

    result = _extractive_fallback("What is supervised learning?", contexts)
    assert result is not None, "FAIL: _extractive_fallback returned None with valid contexts"
    assert "Answer:" in result, f"FAIL: expected 'Answer:' prefix; got: {result!r}"
    assert "Timestamp:" in result, f"FAIL: expected 'Timestamp:' line; got: {result!r}"
    assert "NOT FOUND IN VIDEO" not in result, "FAIL: fallback should not produce NOT FOUND"
    print(f"  ✓ PASS — extractive fallback produced grounded answer")
    print(f"    {result.strip()[:120]}...")


def test_no_fallback_on_empty_contexts():
    """Extractive fallback should return None when contexts are empty."""
    print("\n" + "="*70)
    print("TEST 2: _extractive_fallback() — empty contexts → None")
    print("="*70)

    from video_qa.answer_generator import _extractive_fallback

    result = _extractive_fallback("What is X?", [])
    assert result is None, f"FAIL: expected None for empty contexts; got {result!r}"
    print("  ✓ PASS — returns None when no contexts available")


def test_generate_answer_with_all_llms_failed():
    """
    End-to-end: generate_answer() must NOT return 'NOT FOUND IN VIDEO'
    when all LLMs fail but valid context chunks exist.
    """
    print("\n" + "="*70)
    print("TEST 3: generate_answer() — all LLMs patched to None, contexts exist")
    print("="*70)

    from video_qa.answer_generator import generate_answer

    contexts = [
        {
            "text": "Regularization is a technique used to reduce overfitting in machine learning models by adding a penalty term to the loss function.",
            "start": 459.0,
            "end": 479.0,
            "score": 0.95,
        },
        {
            "text": "L1 regularization adds the absolute value of weights. L2 regularization adds the squared value of weights.",
            "start": 479.0,
            "end": 495.0,
            "score": 0.88,
        },
    ]

    # Patch ALL external LLM providers to return None
    with patch("video_qa.answer_generator.ask_local_llm", return_value=None), \
         patch("video_qa.answer_generator.ask_grok_llm", return_value=None), \
         patch("video_qa.answer_generator.ask_gemini_llm", return_value=None), \
         patch("video_qa.answer_generator.ask_openai_llm", return_value=None), \
         patch("video_qa.answer_generator.ask_huggingface_llm", return_value=None):

        result = generate_answer("What is regularization?", contexts, use_verification=False)

    answer = result.get("answer", "")
    print(f"  Answer: {answer[:120]}")
    print(f"  Timestamp: {result.get('timestamp')}")

    assert answer, "FAIL: answer is empty"
    assert "NOT FOUND IN VIDEO" not in answer.upper(), (
        f"FAIL: should not return NOT FOUND IN VIDEO when contexts exist; got: {answer!r}"
    )
    assert "cannot be answered" not in answer.lower(), (
        f"FAIL: should not return 'cannot be answered' when contexts exist; got: {answer!r}"
    )
    print("  ✓ PASS — got grounded answer even with all LLMs failing")


def test_not_found_only_when_no_contexts():
    """
    generate_answer() MUST return 'NOT FOUND' / 'cannot be answered'
    when contexts list is empty — regardless of LLM state.
    """
    print("\n" + "="*70)
    print("TEST 4: generate_answer() — no contexts → NOT FOUND")
    print("="*70)

    from video_qa.answer_generator import generate_answer

    result = generate_answer("What is regularization?", [], use_verification=False)
    answer = result.get("answer", "")
    print(f"  Answer: {answer[:120]}")

    assert "cannot be answered" in answer.lower() or "not found" in answer.lower(), (
        f"FAIL: expected NOT FOUND / cannot be answered when no contexts; got: {answer!r}"
    )
    print("  ✓ PASS — correctly returns NOT FOUND when no context retrieved")


def test_llm_answer_takes_priority_over_fallback():
    """
    If an LLM (e.g., HuggingFace) returns a valid answer, it must be used
    instead of the extractive fallback.
    """
    print("\n" + "="*70)
    print("TEST 5: LLM answer takes priority over extractive fallback")
    print("="*70)

    from video_qa.answer_generator import generate_answer

    contexts = [
        {
            "text": "Backpropagation is the algorithm used to train neural networks by computing gradients.",
            "start": 120.0,
            "end": 140.0,
            "score": 0.93,
        }
    ]
    llm_answer = "Answer: Backpropagation trains neural networks.\nTimestamp: [02:00 - 02:20]"

    with patch("video_qa.answer_generator.ask_local_llm", return_value=None), \
         patch("video_qa.answer_generator.ask_grok_llm", return_value=None), \
         patch("video_qa.answer_generator.ask_gemini_llm", return_value=None), \
         patch("video_qa.answer_generator.ask_openai_llm", return_value=None), \
         patch("video_qa.answer_generator.ask_huggingface_llm", return_value=llm_answer):

        result = generate_answer("How are neural networks trained?", contexts, use_verification=False)

    answer = result.get("answer", "")
    print(f"  Answer: {answer}")
    assert "backpropagation" in answer.lower(), (
        f"FAIL: LLM answer should be used; got: {answer!r}"
    )
    print("  ✓ PASS — LLM answer used when LLM succeeds")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("EXTRACTIVE FALLBACK TEST SUITE")
    print("="*70)

    tests = [
        test_extractive_fallback_direct,
        test_no_fallback_on_empty_contexts,
        test_generate_answer_with_all_llms_failed,
        test_not_found_only_when_no_contexts,
        test_llm_answer_takes_priority_over_fallback,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"\n  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n  ✗ ERROR in {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*70)
    sys.exit(0 if failed == 0 else 1)
