from __future__ import annotations

from pathlib import Path
from typing import Any


def attach_openclip_embeddings(
    tracklets: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    crop_items = [
        (tracklet, tracklet.get("representative_crop"))
        for tracklet in tracklets
        if tracklet.get("representative_crop")
        and Path(tracklet["representative_crop"]).exists()
    ]
    if not crop_items:
        return {"available": False, "reason": "no_representative_crops"}
    try:
        import open_clip
        import torch
        from PIL import Image
    except ImportError:
        return {"available": False, "reason": "open_clip_not_installed"}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = str(config.get("openclip_model", "ViT-B-32"))
    pretrained = str(
        config.get("openclip_pretrained", "laion2b_s34b_b79k")
    )
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )
    model.eval()
    batch_size = int(config.get("openclip_batch_size", 32))
    with torch.inference_mode():
        for offset in range(0, len(crop_items), batch_size):
            batch = crop_items[offset : offset + batch_size]
            images = torch.stack(
                [preprocess(Image.open(path).convert("RGB")) for _, path in batch]
            ).to(device)
            embeddings = model.encode_image(images)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            for (tracklet, _), embedding in zip(batch, embeddings.cpu()):
                tracklet["clip_embedding"] = embedding.float().tolist()
    del model, images, embeddings
    if device == "cuda":
        torch.cuda.empty_cache()
    return {
        "available": True,
        "model": model_name,
        "pretrained": pretrained,
        "embedded_tracklets": len(crop_items),
    }
