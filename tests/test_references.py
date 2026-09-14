import json
from pathlib import Path
from types import SimpleNamespace

from mara_lab.artifacts import sha256_file
from mara_lab.backends.photomaker_v2 import compile_photomaker_prompt
from mara_lab.backends.reference_fake import FakeReferenceBackend
from mara_lab.config import ReferenceBackendName, load_reference_experiment_config
from mara_lab.disk_offload import map_photomaker_lora_name
from mara_lab.domain import ReferenceShot
from mara_lab.errors import ConfigurationError
from mara_lab.workflows.approve_references import (
    approve_reference_set,
    finalize_reference_set,
    load_reference_set_images,
)
from mara_lab.workflows.references import _sheet_label, run_references

ROOT = Path(__file__).parents[1]


def test_reference_sheet_label_includes_cell_and_seed_index() -> None:
    record = SimpleNamespace(artifact_id="reference-a3-07-seed-31006", seed=31006)

    assert _sheet_label(record) == "A3-07 s31006"


def test_photomaker_prompt_places_trigger_after_class_word() -> None:
    shot = ReferenceShot(
        cell_id="a1",
        view="frontal view",
        expression="neutral expression",
        lighting="indirect window light",
        setting="a plain apartment interior",
        wardrobe="a dark crew-neck shirt",
        framing="close chest-up",
        capture_style="modern phone",
        camera_behavior="natural auto exposure",
    )

    prompt = compile_photomaker_prompt(shot, "img")

    assert "adult woman img," in prompt
    assert prompt.split().count("img,") == 1
    assert "wearing a dark crew-neck shirt" in prompt


def test_photomaker_compact_prompt_preserves_trigger_and_reduces_length() -> None:
    shot = ReferenceShot(
        cell_id="a1",
        view="frontal view",
        expression="neutral expression",
        lighting="indirect window light",
        setting="a plain apartment interior",
        wardrobe="a dark crew-neck shirt",
        framing="close chest-up",
        capture_style="modern phone",
        camera_behavior="natural auto exposure",
    )

    historical = compile_photomaker_prompt(shot, "img", "v1")
    compact = compile_photomaker_prompt(shot, "img", "v2")

    assert "adult woman img," in historical
    assert "adult woman img," in compact
    assert len(compact) < len(historical)


def test_maps_photomaker_lora_names_to_peft() -> None:
    assert map_photomaker_lora_name(
        "unet.down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor.to_q_lora.down.weight"
    ) == ("down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.photomaker.weight")
    assert map_photomaker_lora_name(
        "unet.up_blocks.0.attentions.0.transformer_blocks.0.attn2.processor.to_out_lora.up.weight"
    ) == ("up_blocks.0.attentions.0.transformer_blocks.0.attn2.to_out.0.lora_B.photomaker.weight")


def test_loads_pass_a_reference_matrix() -> None:
    config = load_reference_experiment_config(
        ROOT / "configs" / "experiments" / "mara-references-pass-a-v001.yaml"
    )

    assert config.pass_id == "a"
    assert len(config.cells) == 6
    assert config.generation.seeds.count == 8
    assert config.maximum_reference_images == 1
    assert config.prompt_template_version == "v1"
    assert config.conditioner.revision == "f5a1e5155dc02166253fa7e29d13519f5ba22eac"
    assert config.reference_images[0].name == "master.png"


def test_loads_72_image_dataset_candidate_matrix() -> None:
    config = load_reference_experiment_config(
        ROOT / "configs" / "experiments" / "mara-dataset-candidates-v001.yaml"
    )

    assert config.pass_id == "c"
    assert len(config.cells) == 18
    assert config.generation.seeds.count == 4
    assert len(config.cells) * config.generation.seeds.count == 72
    assert config.minimum_reference_images == 7
    assert config.maximum_reference_images == 7
    assert config.prompt_template_version == "v2"


def test_reference_workflow_records_lineage(tmp_path: Path) -> None:
    config = load_reference_experiment_config(
        ROOT / "configs" / "experiments" / "mara-references-pass-a-v001.yaml"
    )
    config = config.model_copy(
        update={
            "conditioner": config.conditioner.model_copy(
                update={"backend": ReferenceBackendName.FAKE}
            ),
            "output": config.output.model_copy(update={"root": tmp_path}),
        }
    )

    summary = run_references(
        config,
        image_limit=3,
        run_id="reference-test",
        backend_factory=FakeReferenceBackend,
    )

    assert summary.generated_count == 3
    assert summary.contact_sheet_path.is_file()
    records = [
        json.loads(line) for line in summary.manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["cell_id"] for record in records] == ["a1", "a1", "a1"]
    assert [record["seed"] for record in records] == [31000, 31001, 31002]
    assert all(record["source_references"][0]["sha256"] for record in records)
    assert json.loads((summary.run_dir / "status.json").read_text())["state"] == "completed"


def test_reference_workflow_resumes_from_verified_manifest_prefix(tmp_path: Path) -> None:
    config = load_reference_experiment_config(
        ROOT / "configs" / "experiments" / "mara-references-pass-a-v001.yaml"
    )
    config = config.model_copy(
        update={
            "conditioner": config.conditioner.model_copy(
                update={"backend": ReferenceBackendName.FAKE}
            ),
            "output": config.output.model_copy(update={"root": tmp_path}),
        }
    )
    initial = run_references(
        config,
        image_limit=3,
        run_id="reference-resume-test",
        backend_factory=FakeReferenceBackend,
    )
    original_hashes = {
        path.name: sha256_file(path) for path in sorted((initial.run_dir / "outputs").glob("*.png"))
    }
    status_path = initial.run_dir / "status.json"
    interrupted_status = json.loads(status_path.read_text(encoding="utf-8"))
    interrupted_status.update(
        {
            "state": "failed",
            "finished_at": "2026-01-01T00:00:00+00:00",
            "error": "simulated interruption",
        }
    )
    status_path.write_text(json.dumps(interrupted_status), encoding="utf-8")

    resumed = run_references(
        config,
        image_limit=5,
        run_id="reference-resume-test",
        resume=True,
        backend_factory=FakeReferenceBackend,
    )

    assert resumed.generated_count == 5
    assert all(
        sha256_file(resumed.run_dir / "outputs" / name) == digest
        for name, digest in original_hashes.items()
    )
    assert len(resumed.manifest_path.read_text(encoding="utf-8").splitlines()) == 5
    assert (resumed.run_dir / "resume-events.jsonl").is_file()
    final_status = json.loads(status_path.read_text(encoding="utf-8"))
    assert final_status["state"] == "completed"
    assert final_status["generated_count"] == 5
    assert final_status["error"] is None


def test_approved_pass_a_set_can_seed_pass_b(tmp_path: Path) -> None:
    pass_a = load_reference_experiment_config(
        ROOT / "configs" / "experiments" / "mara-references-pass-a-v001.yaml"
    )
    pass_a = pass_a.model_copy(
        update={
            "conditioner": pass_a.conditioner.model_copy(
                update={"backend": ReferenceBackendName.FAKE}
            ),
            "output": pass_a.output.model_copy(update={"root": tmp_path / "experiments"}),
        }
    )
    source_run = run_references(
        pass_a,
        image_limit=3,
        run_id="pass-a",
        backend_factory=FakeReferenceBackend,
    )
    approved = approve_reference_set(
        source_run.run_dir,
        ["reference-a1-01-seed-31000", "reference-a1-03-seed-31002"],
        characters_root=tmp_path / "characters",
    )

    images = load_reference_set_images(approved.descriptor_path, character_id="mara")
    assert len(images) == 3
    assert images[0].name == "master.png"

    pass_b = load_reference_experiment_config(
        ROOT / "configs" / "experiments" / "mara-references-pass-b-v001.yaml"
    )
    assert len(pass_b.cells) * pass_b.generation.seeds.count == 32
    pass_b = pass_b.model_copy(
        update={
            "reference_images": images,
            "conditioner": pass_b.conditioner.model_copy(
                update={"backend": ReferenceBackendName.FAKE}
            ),
            "output": pass_b.output.model_copy(update={"root": tmp_path / "experiments"}),
        }
    )
    pass_b_run = run_references(
        pass_b,
        image_limit=1,
        run_id="pass-b",
        backend_factory=FakeReferenceBackend,
    )
    assert pass_b_run.generated_count == 1


def test_pass_b_rejects_master_only(tmp_path: Path) -> None:
    config = load_reference_experiment_config(
        ROOT / "configs" / "experiments" / "mara-references-pass-b-v001.yaml"
    )
    config = config.model_copy(
        update={
            "conditioner": config.conditioner.model_copy(
                update={"backend": ReferenceBackendName.FAKE}
            ),
            "output": config.output.model_copy(update={"root": tmp_path}),
        }
    )

    try:
        run_references(config, image_limit=1, backend_factory=FakeReferenceBackend)
    except ConfigurationError as exc:
        assert "requires at least 3 reference images" in str(exc)
    else:
        raise AssertionError("Pass B accepted a master-only reference set")


def test_final_reference_set_can_seed_dataset_candidates(tmp_path: Path) -> None:
    pass_a = load_reference_experiment_config(
        ROOT / "configs" / "experiments" / "mara-references-pass-a-v001.yaml"
    )
    pass_a = pass_a.model_copy(
        update={
            "conditioner": pass_a.conditioner.model_copy(
                update={"backend": ReferenceBackendName.FAKE}
            ),
            "output": pass_a.output.model_copy(update={"root": tmp_path / "experiments"}),
        }
    )
    pass_a_run = run_references(
        pass_a,
        image_limit=4,
        run_id="pass-a-final",
        backend_factory=FakeReferenceBackend,
    )
    interim = approve_reference_set(
        pass_a_run.run_dir,
        [
            "reference-a1-01-seed-31000",
            "reference-a1-02-seed-31001",
            "reference-a1-03-seed-31002",
            "reference-a1-04-seed-31003",
        ],
        characters_root=tmp_path / "characters",
    )

    pass_b = load_reference_experiment_config(
        ROOT / "configs" / "experiments" / "mara-references-pass-b-v001.yaml"
    )
    pass_b = pass_b.model_copy(
        update={
            "reference_images": interim.image_paths,
            "conditioner": pass_b.conditioner.model_copy(
                update={"backend": ReferenceBackendName.FAKE}
            ),
            "output": pass_b.output.model_copy(update={"root": tmp_path / "experiments"}),
        }
    )
    pass_b_run = run_references(
        pass_b,
        image_limit=2,
        run_id="pass-b-final",
        backend_factory=FakeReferenceBackend,
    )
    final = finalize_reference_set(
        interim.descriptor_path,
        pass_b_run.run_dir,
        ["reference-b1-01-seed-32000", "reference-b1-02-seed-32001"],
        characters_root=tmp_path / "characters",
    )
    assert len(final.image_paths) == 7

    dataset_candidates = load_reference_experiment_config(
        ROOT / "configs" / "experiments" / "mara-dataset-candidates-v001.yaml"
    )
    dataset_candidates = dataset_candidates.model_copy(
        update={
            "reference_images": final.image_paths,
            "conditioner": dataset_candidates.conditioner.model_copy(
                update={"backend": ReferenceBackendName.FAKE}
            ),
            "output": dataset_candidates.output.model_copy(
                update={"root": tmp_path / "experiments"}
            ),
        }
    )
    candidate_run = run_references(
        dataset_candidates,
        image_limit=1,
        run_id="dataset-candidates",
        backend_factory=FakeReferenceBackend,
    )
    assert candidate_run.generated_count == 1
