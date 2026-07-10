from typing import Any

from model_client.types import ModelRequestError

DEFAULT_IMAGE_MIME_TYPE = "image/jpeg"


def build_frame_grid_context(
    *,
    frame_count: int,
    cols: int,
    rows: int,
    empty_cells: int,
    width_px: int,
    height_px: int,
) -> str:
    capacity = cols * rows
    return (
        "You are given ONE base64-encoded JPEG image. "
        f"It is a single montage that tiles {frame_count} video frame(s) into a "
        f"{rows}x{cols} grid (capacity {capacity} cells), read left-to-right then "
        "top-to-bottom in chronological order. "
        f"The image is {width_px}x{height_px} pixels. "
        f"{empty_cells} cell(s) contain no frame and are filled with solid black; "
        "individual frames may also carry black letterbox bars to preserve aspect "
        "ratio. Treat any all-black cell or black bar as padding, not as video "
        "content, and never describe it.\n\n"
    )


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


def build_frame_grid_messages(
    frame_grid_base64: str,
    system_prompt: str,
    user_prompt: str,
    *,
    frame_count: int,
    cols: int,
    rows: int,
    empty_cells: int,
    width_px: int,
    height_px: int,
    image_mime_type: str = DEFAULT_IMAGE_MIME_TYPE,
) -> list[dict[str, Any]]:
    context = build_frame_grid_context(
        frame_count=frame_count,
        cols=cols,
        rows=rows,
        empty_cells=empty_cells,
        width_px=width_px,
        height_px=height_px,
    )
    return build_image_messages(
        [frame_grid_base64],
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
