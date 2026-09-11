import json
from pathlib import Path

import pytest
from PIL import Image

from mara_lab.backends.fake import FakeImageBackend
from mara_lab.config import BackendName, load_experiment_config
from mara_lab.domain import GenerationRequest, GenerationResult, RuntimeMetrics
from mara_lab.errors import MemoryBudgetExceededError
from mara_lab.workflows.candidates import run_candidates
from mara_lab.workflows.reproduce import reproduce_candidate

ROOT = Path(__file__).parents[1]


def fake_config(tmp_path: Path):
    config = load_experiment_config(ROOT / "configs" / "experiments" / "mara-candidates-v001.yaml")
    model = config.model.model_copy(
        update={
            "backend": BackendName.FAKE,
            "model_id": "mara-lab/fake",
            "revision": "fake-v1",
            "resolved_revision": "fake-v1",
        }
    )
    output = config.output.model_copy(update={"root": tmp_path})
    return config.model_copy(update={"model": model, "output": output})


def test_candidate_run_writes_reproducible_artifacts(tmp_path: Path) -> None:
    config = fake_config(tmp_path)
    summary = run_candidates(config, seed_limit=3, run_id="test-run")

    assert summary.generated_count == 3
    assert summary.manifest_path.exists()
    assert summary.contact_sheet_path.exists()
    assert (summary.run_dir / "resolved-config.yaml").exists()
    assert (summary.run_dir / "environment.json").exists()
    assert (summary.run_dir / "status.json").read_text(encoding="utf-8").find(
        '"state": "completed"'
    ) >= 0
    assert len(summary.manifest_path.read_text(encoding="utf-8").splitlines()) == 3
    records = [json.loads(line) for line in summary.manifest_path.read_text().splitlines()]
    assert [record["generation"]["identity_variant"] for record in records] == [1, 2, 3]

    with Image.open(summary.contact_sheet_path) as contact_sheet:
        assert contact_sheet.width > 0
        assert contact_sheet.height > 0

    reproduction = reproduce_candidate(summary.run_dir, "candidate-0001-seed-11000", strict=True)
    assert reproduction.exact_match
    assert reproduction.expected_sha256 == reproduction.actual_sha256


class OverBudgetBackend(FakeImageBackend):
    def generate(self, request: GenerationRequest) -> GenerationResult:
        result = super().generate(request)
        return GenerationResult(
            image=result.image,
            metrics=RuntimeMetrics(
                wall_seconds=result.metrics.wall_seconds,
                max_allocated_vram_gib=10.6,
                max_reserved_vram_gib=10.6,
            ),
        )


def test_candidate_run_preserves_partial_artifacts_on_memory_failure(tmp_path: Path) -> None:
    config = fake_config(tmp_path)

    with pytest.raises(MemoryBudgetExceededError, match="exceeded VRAM budget"):
        run_candidates(
            config,
            seed_limit=2,
            run_id="over-budget",
            backend_factory=lambda resolved: OverBudgetBackend(resolved.model, resolved.compute),
        )

    run_dir = tmp_path / "over-budget"
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    manifest_lines = (run_dir / "outputs" / "manifest.jsonl").read_text().splitlines()

    assert status["state"] == "failed"
    assert status["generated_count"] == 1
    assert "MemoryBudgetExceededError" in status["error"]
    assert len(manifest_lines) == 1
    assert (run_dir / "contact-sheets" / "candidates.png").exists()
