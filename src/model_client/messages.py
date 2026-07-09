from typing import Any

from model_client.types import ModelRequestError

DEFAULT_IMAGE_MIME_TYPE = "image/jpeg"
FRAME_GRID_PROMPT_PREFIX = (
    "The attached image is a grid of video frames ordered left-to-right, "
    "top-to-bottom. Analyze it as a temporal sequence.\n\n"
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
    image_mime_type: str = DEFAULT_IMAGE_MIME_TYPE,
) -> list[dict[str, Any]]:
    return build_image_messages(
        [frame_grid_base64],
        system_prompt,
        f"{FRAME_GRID_PROMPT_PREFIX}{user_prompt}",
        image_mime_type=image_mime_type,
    )


def normalize_image_data_url(image_base64: str, image_mime_type: str) -> str:
    value = image_base64.strip()
    if not value:
        raise ModelRequestError("image base64 value cannot be empty")

    if value.startswith("data:"):
        return value

    return f"data:{image_mime_type};base64,{value}"
