from typing import Any

from model_client.types import ModelRequestError

DEFAULT_IMAGE_MIME_TYPE = "image/jpeg"


def build_frame_grids_context(grids_meta: list[dict[str, int]], *, image_count: int) -> str:
    lines = [
        f"You are given {image_count} base64-encoded JPEG image(s), in chronological order "
        "(image 1 first). Together they cover one video.",
    ]
    for index, meta in enumerate(grids_meta, start=1):
        lines.append(
            f"Image {index}: {meta['frame_count']} frame(s) in a "
            f"{meta['rows']}x{meta['cols']} grid ({meta['width_px']}x{meta['height_px']} px), "
            f"{meta['empty_cells']} empty cell(s) filled solid black."
        )
    lines.append(
        "Each image tiles frames left-to-right then top-to-bottom in chronological order. "
        "Individual frames may also carry black letterbox bars to preserve aspect ratio. "
        "Treat any all-black cell or black bar as padding, not as video content, and never "
        "describe it."
    )
    return "\n".join(lines) + "\n\n"


def build_text_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_image_messages(
    images_base64: list[str],
    system_prompt: str,
    user_prompt: str,
    *,
    image_mime_type: str = DEFAULT_IMAGE_MIME_TYPE,
) -> list[dict[str, Any]]:
    if not images_base64:
        raise ModelRequestError("images_base64 must contain at least one image")

    content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for image_base64 in images_base64:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": normalize_image_data_url(image_base64, image_mime_type),
                },
            }
        )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def build_frame_grids_messages(
    grids_base64: list[str],
    system_prompt: str,
    user_prompt: str,
    *,
    grids_meta: list[dict[str, int]],
    image_mime_type: str = DEFAULT_IMAGE_MIME_TYPE,
) -> list[dict[str, Any]]:
    if len(grids_base64) != len(grids_meta):
        raise ModelRequestError("grids_base64 and grids_meta must have the same length")
    if not grids_base64:
        raise ModelRequestError("grids_base64 must contain at least one image")

    context = build_frame_grids_context(grids_meta, image_count=len(grids_base64))
    return build_image_messages(
        grids_base64,
        system_prompt,
        f"{context}{user_prompt}",
        image_mime_type=image_mime_type,
    )


def normalize_image_data_url(image_base64: str, image_mime_type: str) -> str:
    value = image_base64.strip()
    if not value:
        raise ModelRequestError("image base64 value cannot be empty")

    if value.startswith("data:"):
        return value

    return f"data:{image_mime_type};base64,{value}"
