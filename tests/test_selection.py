import json
from pathlib import Path

import pytest
import yaml

from mara_lab.errors import ConfigurationError
from mara_lab.workflows.candidates import run_candidates
from mara_lab.workflows.select import select_canon
from tests.test_workflow import fake_config


def test_select_canon_archives_images_and_lineage(tmp_path: Path) -> None:
    source_config = fake_config(tmp_path / "experiments")
    run = run_candidates(source_config, seed_limit=3, run_id="source-run")

    summary = select_canon(
        run.run_dir,
        "candidate-0003-seed-11002",
        "candidate-0002-seed-11001",
        canon_root=tmp_path / "characters",
    )

    assert (
        summary.master_path.read_bytes()
        == (run.run_dir / "outputs" / "candidate-0003-seed-11002.png").read_bytes()
    )
    assert (
        summary.backup_path.read_bytes()
        == (run.run_dir / "outputs" / "candidate-0002-seed-11001.png").read_bytes()
    )

    records = [
        json.loads(line) for line in summary.manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["role"] for record in records] == ["master", "backup"]
    assert [record["source_artifact"]["artifact_id"] for record in records] == [
        "candidate-0003-seed-11002",
        "candidate-0002-seed-11001",
    ]

    character = yaml.safe_load(
        (summary.character_dir / "character.yaml").read_text(encoding="utf-8")
    )
    assert character["canonical"]["master"] == "canonical/v001/master.png"
    assert character["canonical"]["backup"] == "canonical/v001/backup.png"

    environment = json.loads(
        (summary.master_path.parent / "source-environment.json").read_text(encoding="utf-8")
    )
    assert "python_executable" not in environment

    with pytest.raises(ConfigurationError, match="canon already exists"):
        select_canon(
            run.run_dir,
            "candidate-0003-seed-11002",
            "candidate-0002-seed-11001",
            canon_root=tmp_path / "characters",
        )


def test_select_canon_rejects_duplicate_selection(tmp_path: Path) -> None:
    source_config = fake_config(tmp_path / "experiments")
    run = run_candidates(source_config, seed_limit=1, run_id="source-run")

    with pytest.raises(ConfigurationError, match="must be different"):
        select_canon(
            run.run_dir,
            "candidate-0001-seed-11000",
            "candidate-0001-seed-11000",
            canon_root=tmp_path / "characters",
        )


def test_select_canon_rejects_modified_source_image(tmp_path: Path) -> None:
    source_config = fake_config(tmp_path / "experiments")
    run = run_candidates(source_config, seed_limit=2, run_id="source-run")
    source_image = run.run_dir / "outputs" / "candidate-0002-seed-11001.png"
    source_image.write_bytes(b"modified after generation")

    with pytest.raises(ConfigurationError, match="source hash mismatch"):
        select_canon(
            run.run_dir,
            "candidate-0002-seed-11001",
            "candidate-0001-seed-11000",
            canon_root=tmp_path / "characters",
        )
