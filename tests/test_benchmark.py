import json
import struct
from pathlib import Path

from mara_lab.artifacts import append_jsonl, atomic_write_json, sha256_file, utc_now
from mara_lab.backends.fake import FakeImageBackend
from mara_lab.config import load_benchmark_experiment_config
from mara_lab.manifests import AdapterRecord, RunStatus
from mara_lab.workflows.benchmark import run_benchmark, validate_adapter_for_benchmark

ROOT = Path(__file__).parents[1]


def _adapter(tmp_path: Path) -> tuple[Path, AdapterRecord]:
    config = load_benchmark_experiment_config(
        ROOT / "configs" / "benchmarks" / "mara-lora-sentinels-v001.yaml"
    )
    run_dir = tmp_path / "training-run"
    adapter_path = run_dir / "adapters" / "mara-test.safetensors"
    adapter_path.parent.mkdir(parents=True)
    header = json.dumps(
        {"lora_unet_test": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    ).encode("utf-8")
    adapter_path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0\0\0")
    record = AdapterRecord(
        adapter_id="mara-test",
        character_id="mara",
        adapter_path="adapters/mara-test.safetensors",
        sha256=sha256_file(adapter_path),
        base_model_id=config.model.model_id,
        base_revision=config.model.resolved_revision or config.model.revision,
        base_weight_sha256=config.model.weight_sha256,
        trainer_repository="https://example.invalid/sd-scripts.git",
        trainer_revision="a" * 40,
        dataset_manifest_sha256="b" * 64,
        train_steps=1,
        network_dim=16,
        network_alpha=16,
        tensor_count=2,
        up_tensor_count=1,
        nonzero_up_tensor_count=1,
        max_abs_weight=0.01,
        max_reserved_vram_gib=1.0,
        created_at=utc_now(),
    )
    append_jsonl(adapter_path.parent / "manifest.jsonl", record.model_dump(mode="json"))
    status = RunStatus(
        state="completed",
        experiment_id="training-test",
        run_id=run_dir.name,
        generated_count=1,
        started_at=utc_now(),
        finished_at=utc_now(),
    )
    atomic_write_json(run_dir / "status.json", status.model_dump(mode="json"))
    return adapter_path, record


def test_validates_and_renders_traced_adapter(tmp_path: Path) -> None:
    adapter_path, expected_record = _adapter(tmp_path)
    config = load_benchmark_experiment_config(
        ROOT / "configs" / "benchmarks" / "mara-lora-sentinels-v001.yaml"
    )
    config = config.model_copy(
        update={"output": config.output.model_copy(update={"root": tmp_path / "experiments"})}
    )

    assert validate_adapter_for_benchmark(adapter_path, config) == expected_record

    def fake_backend(resolved, _adapter_path, _adapter_weight):
        return FakeImageBackend(resolved.model, resolved.compute)

    summary = run_benchmark(
        config,
        adapter_path,
        case_limit=2,
        run_id="benchmark-test",
        backend_factory=fake_backend,
    )

    assert summary.generated_count == 2
    assert summary.contact_sheet_path.is_file()
    rows = [
        json.loads(line) for line in summary.manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["case_id"] for row in rows] == [
        "bedroom-morning",
        "bathroom-fluorescent",
    ]
    assert all(row["adapter"]["sha256"] == expected_record.sha256 for row in rows)
