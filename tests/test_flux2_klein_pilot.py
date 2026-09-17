from __future__ import annotations

import sys
from pathlib import Path

from mara_lab.flux2_klein import (
    DEFAULT_FP8_TRANSFORMER_ROOT,
    DEFAULT_MODEL_ROOT,
    DEFAULT_TEXT_ENCODER_ROOT,
)
from scripts import flux2_klein_pilot


def test_pilot_matrix_has_unique_reproducible_cases() -> None:
    assert len(flux2_klein_pilot.CASES) == 10
    assert len({case.case_id for case in flux2_klein_pilot.CASES}) == 10
    assert len({case.seed for case in flux2_klein_pilot.CASES}) == 10
    assert all("same fictional adult woman" in case.prompt for case in flux2_klein_pilot.CASES)


def test_body_expression_gate_has_unique_reproducible_cases() -> None:
    cases = flux2_klein_pilot.BODY_EXPRESSION_CASES

    assert len(cases) == 24
    assert len({case.case_id for case in cases}) == 24
    assert len({case.seed for case in cases}) == 24
    assert not {case.case_id for case in cases} & {case.case_id for case in flux2_klein_pilot.CASES}
    assert not {case.seed for case in cases} & {case.seed for case in flux2_klein_pilot.CASES}
    assert all("same fictional adult woman" in case.prompt for case in cases)


def test_embedding_cache_key_changes_with_prompt_and_length(tmp_path: Path) -> None:
    first = flux2_klein_pilot.embedding_cache_path(tmp_path, "first", 256)
    repeated = flux2_klein_pilot.embedding_cache_path(tmp_path, "first", 256)
    different_prompt = flux2_klein_pilot.embedding_cache_path(tmp_path, "second", 256)
    different_length = flux2_klein_pilot.embedding_cache_path(tmp_path, "first", 512)

    assert first == repeated
    assert len({first, different_prompt, different_length}) == 3
    assert first.parent == tmp_path / "prompt-embeddings"


def test_cli_defaults_use_pinned_cache_layout(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["flux2_klein_pilot.py", "--run-id", "test-run"])

    args = flux2_klein_pilot.parse_args()

    assert args.model_root == DEFAULT_MODEL_ROOT
    assert args.text_encoder_root == DEFAULT_TEXT_ENCODER_ROOT
    assert args.fp8_transformer_root == DEFAULT_FP8_TRANSFORMER_ROOT
    assert args.suite == "everyday"
    assert args.limit == 1
