from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract one InsightFace identity embedding in an isolated process."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    return parser


def extract_embedding(
    image_path: Path,
    output_path: Path,
    *,
    root: Path,
    model: str,
    provider: str,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str((Path(".cache") / "matplotlib").resolve()))

    try:
        import cv2
        import numpy as np
        from photomaker import FaceAnalysis2, analyze_faces
    except ImportError as exc:
        raise RuntimeError(
            "PhotoMaker reference dependencies are missing. Run `uv sync --extra reference`."
        ) from exc

    image = cv2.imread(str(image_path.resolve()))
    if image is None:
        raise RuntimeError(f"cannot decode reference image: {image_path}")

    detector = FaceAnalysis2(
        name=model,
        root=str(root.resolve()),
        providers=[provider],
        allowed_modules=["detection", "recognition"],
    )
    detector.prepare(ctx_id=-1, det_size=(640, 640))
    faces = analyze_faces(detector, image)
    if len(faces) != 1:
        raise RuntimeError(
            f"reference image must contain exactly one detectable face; found {len(faces)} "
            f"in {image_path}"
        )

    embedding = np.asarray(faces[0]["embedding"], dtype=np.float32)[None, :]
    if embedding.shape != (1, 512) or not np.isfinite(embedding).all():
        raise RuntimeError(
            f"InsightFace returned an invalid embedding with shape {embedding.shape}"
        )

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, embedding, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = _parser().parse_args()
    try:
        extract_embedding(
            args.image,
            args.output,
            root=args.root,
            model=args.model,
            provider=args.provider,
        )
    except Exception as exc:
        print(f"face embedding failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
