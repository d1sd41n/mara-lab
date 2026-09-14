from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageDraw

from mara_lab.artifacts import save_png_atomic


class ContactSheetRecord(Protocol):
    artifact_id: str
    sequence: int
    seed: int
    width: int
    height: int
    image_path: str


def create_contact_sheet(
    run_dir: Path,
    records: Sequence[ContactSheetRecord],
    destination: Path,
    columns: int,
    thumbnail_width: int,
    labeler: Callable[[ContactSheetRecord], str] | None = None,
) -> None:
    if not records:
        return

    margin = 12
    label_height = 28
    thumbnails: list[tuple[ContactSheetRecord, Image.Image]] = []
    max_thumbnail_height = 0
    for record in records:
        source = run_dir / Path(record.image_path)
        with Image.open(source) as opened:
            thumbnail = opened.convert("RGB")
            target_height = max(1, round(thumbnail_width * record.height / record.width))
            thumbnail.thumbnail((thumbnail_width, target_height), Image.Resampling.LANCZOS)
            thumbnail = thumbnail.copy()
        max_thumbnail_height = max(max_thumbnail_height, thumbnail.height)
        thumbnails.append((record, thumbnail))

    rows = math.ceil(len(thumbnails) / columns)
    cell_width = thumbnail_width + margin
    cell_height = max_thumbnail_height + label_height + margin
    sheet_width = columns * cell_width + margin
    sheet_height = rows * cell_height + margin
    sheet = Image.new("RGB", (sheet_width, sheet_height), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)

    for index, (record, thumbnail) in enumerate(thumbnails):
        row, column = divmod(index, columns)
        x = margin + column * cell_width
        y = margin + row * cell_height
        sheet.paste(thumbnail, (x, y))
        label = (
            labeler(record)
            if labeler is not None
            else f"#{record.sequence:04d}  seed={record.seed}"
        )
        draw.text(
            (x, y + max_thumbnail_height + 6),
            label,
            fill=(20, 20, 20),
        )

    save_png_atomic(sheet, destination)
