from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from mara_lab.artifacts import atomic_write_json, sha256_file


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if vector.shape != (512,) or not np.isfinite(vector).all() or norm == 0:
        raise ValueError("expected one finite, non-zero 512-dimensional face embedding")
    return vector / norm


def build_reference_centroid(embeddings: list[np.ndarray]) -> np.ndarray:
    if not embeddings:
        raise ValueError("at least one reference embedding is required")
    normalized = np.stack([normalize_embedding(embedding) for embedding in embeddings])
    return normalize_embedding(normalized.mean(axis=0))


def cosine_similarity(embedding: np.ndarray, centroid: np.ndarray) -> float:
    return float(np.dot(normalize_embedding(embedding), normalize_embedding(centroid)))


def read_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("image_path"), str):
                raise ValueError(f"invalid record on manifest line {line_number}")
            records.append(record)
    if not records:
        raise ValueError(f"manifest has no records: {path}")
    return records


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def load_detector(root: Path, model: str, provider: str):
    os.environ.setdefault("MPLCONFIGDIR", str((Path(".cache") / "matplotlib").resolve()))
    try:
        from photomaker import FaceAnalysis2
    except ImportError as exc:
        raise RuntimeError(
            "PhotoMaker reference dependencies are missing. Run `uv sync --extra reference`."
        ) from exc

    detector = FaceAnalysis2(
        name=model,
        root=str(root.resolve()),
        providers=[provider],
        allowed_modules=["detection", "recognition"],
    )
    detector.prepare(ctx_id=-1, det_size=(640, 640))
    return detector


def extract_face(detector, path: Path) -> tuple[np.ndarray | None, int, list[float] | None]:
    import cv2
    from photomaker import analyze_faces

    image = cv2.imread(str(path.resolve()))
    if image is None:
        raise RuntimeError(f"cannot decode image: {path}")
    faces = analyze_faces(detector, image)
    if len(faces) != 1:
        return None, len(faces), None
    face = faces[0]
    embedding = np.asarray(face["embedding"], dtype=np.float32)
    bbox = np.asarray(face["bbox"], dtype=np.float32).round(2).tolist()
    return embedding, 1, bbox


def score_run(args: argparse.Namespace) -> Path:
    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "outputs" / "manifest.jsonl"
    records = read_manifest(manifest_path)
    references = [path.resolve() for path in args.reference]
    missing = [str(path) for path in references if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing identity references: {', '.join(missing)}")

    detector = load_detector(args.insightface_root, args.model, args.provider)
    reference_embeddings: list[np.ndarray] = []
    reference_records: list[dict[str, Any]] = []
    for path in references:
        embedding, face_count, bbox = extract_face(detector, path)
        if embedding is None:
            raise RuntimeError(
                f"identity reference must contain exactly one detectable face; found "
                f"{face_count} in {path}"
            )
        reference_embeddings.append(embedding)
        reference_records.append(
            {
                "path": portable_path(path),
                "sha256": sha256_file(path),
                "bbox": bbox,
            }
        )

    centroid = build_reference_centroid(reference_embeddings)
    if len(reference_embeddings) == 1:
        reference_pair_cosine = None
    else:
        pair_scores = [
            cosine_similarity(reference_embeddings[left], reference_embeddings[right])
            for left in range(len(reference_embeddings))
            for right in range(left + 1, len(reference_embeddings))
        ]
        reference_pair_cosine = round(float(np.mean(pair_scores)), 6)

    results: list[dict[str, Any]] = []
    detected_scores: list[float] = []
    for index, record in enumerate(records, start=1):
        image_path = run_dir / record["image_path"]
        print(f"Scoring {index}/{len(records)}: {record['artifact_id']}", flush=True)
        embedding, face_count, bbox = extract_face(detector, image_path)
        score = None if embedding is None else cosine_similarity(embedding, centroid)
        if score is not None:
            detected_scores.append(score)
        results.append(
            {
                "artifact_id": record["artifact_id"],
                "sequence": record.get("sequence"),
                "image_path": record["image_path"],
                "image_sha256": sha256_file(image_path),
                "faces_detected": face_count,
                "face_bbox": bbox,
                "cosine_to_reference_centroid": None if score is None else round(score, 6),
            }
        )

    score_array = np.asarray(detected_scores, dtype=np.float32)
    summary = {
        "generated_images": len(records),
        "single_face_detections": len(detected_scores),
        "detection_rate": round(len(detected_scores) / len(records), 6),
        "cosine_min": None if not detected_scores else round(float(score_array.min()), 6),
        "cosine_median": None if not detected_scores else round(float(np.median(score_array)), 6),
        "cosine_mean": None if not detected_scores else round(float(score_array.mean()), 6),
        "cosine_max": None if not detected_scores else round(float(score_array.max()), 6),
    }
    output_path = args.output.resolve() if args.output else run_dir / "identity-evaluation.json"
    atomic_write_json(
        output_path,
        {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "method": {
                "model": args.model,
                "provider": args.provider,
                "metric": "cosine similarity to the normalized reference centroid",
                "threshold": None,
                "interpretation": "Supporting measurement only; no calibrated pass threshold.",
            },
            "run_dir": portable_path(run_dir),
            "manifest_sha256": sha256_file(manifest_path),
            "references": reference_records,
            "reference_pair_cosine_mean": reference_pair_cosine,
            "summary": summary,
            "results": results,
        },
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Identity report: {output_path}", flush=True)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score faces in a FLUX.2 run against canonical identity references."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--insightface-root", type=Path, default=Path(".cache/insightface"))
    parser.add_argument("--model", default="buffalo_l")
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument(
        "--reference",
        type=Path,
        action="append",
        default=None,
        help="Repeat for each identity reference image.",
    )
    args = parser.parse_args()
    if args.reference is None:
        args.reference = [
            Path("characters/mara/canonical/v001/master.png"),
            Path("characters/mara/canonical/v001/backup.png"),
        ]
    return args


if __name__ == "__main__":
    score_run(parse_args())
