from __future__ import annotations

from PIL import Image, ImageDraw

from .schema import EntityObservation


def draw_entity_boxes(frame, observations: list[EntityObservation]) -> Image.Image:
    """Draw entity IDs on a frame.

    This is intentionally not wired into ingestion yet; the first MVP injects
    entity memory as text. The helper is here for the next step: ID-overlaid
    frames for stronger VLM grounding.
    """
    image = frame if isinstance(frame, Image.Image) else Image.fromarray(frame)
    image = image.copy()
    draw = ImageDraw.Draw(image)
    for obs in observations:
        x1, y1, x2, y2 = obs.bbox
        label = obs.tracklet_id
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(0, y1 - 14)), label, fill="red")
    return image
