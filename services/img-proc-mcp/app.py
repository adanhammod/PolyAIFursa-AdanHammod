import base64
import io
import os

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

MCP_PORT = int(os.environ.get("FASTMCP_PORT", 9000))

mcp = FastMCP(
    "img-proc",
    host="0.0.0.0",
    port=MCP_PORT,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[
            "img-proc-mcp:9000",
            "localhost:9000",
            "127.0.0.1:9000",
            "host.docker.internal:9000",
        ],
        allowed_origins=["*"],
    ),
)


def _decode(image_b64: str) -> Image.Image:
    data = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(data))


def _encode(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@mcp.tool()
def rotate(image_b64: str, angle: float) -> str:
    img = _decode(image_b64)
    return _encode(img.rotate(angle, expand=True))


@mcp.tool()
def flip(image_b64: str, direction: str = "horizontal") -> str:
    img = _decode(image_b64)
    if direction == "horizontal":
        return _encode(ImageOps.mirror(img))
    elif direction == "vertical":
        return _encode(ImageOps.flip(img))
    else:
        raise ValueError(
            f"Unsupported direction: {direction!r}. Use 'horizontal' or 'vertical'."
        )


@mcp.tool()
def blur(image_b64: str, radius: float = 2.0) -> str:
    img = _decode(image_b64)
    return _encode(img.filter(ImageFilter.GaussianBlur(radius=radius)))


@mcp.tool()
def resize(image_b64: str, width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Width and height must be positive, got width={width}, height={height}."
        )
    img = _decode(image_b64)
    return _encode(img.resize((width, height)))


@mcp.tool()
def crop(image_b64: str, left: int, top: int, right: int, bottom: int) -> str:
    img = _decode(image_b64)
    w, h = img.size
    if left < 0 or top < 0 or right > w or bottom > h or left >= right or top >= bottom:
        raise ValueError(
            f"Invalid crop box ({left}, {top}, {right}, {bottom}) for image of size {w}x{h}."
        )
    return _encode(img.crop((left, top, right, bottom)))


@mcp.tool()
def add_noise(image_b64: str, amount: float = 0.02) -> str:
    if not (0 <= amount <= 1):
        raise ValueError(f"amount must be between 0 and 1, got {amount}.")
    img = _decode(image_b64).convert("RGB")
    arr = np.array(img)
    rng = np.random.default_rng()
    mask = rng.random(arr.shape[:2])
    arr[mask < amount / 2] = 0
    arr[mask > 1 - amount / 2] = 255
    return _encode(Image.fromarray(arr))


@mcp.tool()
def replace_region(
    original_image_b64: str,
    processed_region_b64: str,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> str:
    """Paste a processed region onto the original image, centered on the original bounding box.

    The processed region is used at its natural size (no scaling).
    Any part that extends beyond the original image boundary is clipped.
    Returns the full composite image as base64 PNG.
    """
    original = _decode(original_image_b64).convert("RGB")
    region = _decode(processed_region_b64).convert("RGB")

    orig_w, orig_h = original.size
    region_w, region_h = region.size

    cx = (left + right) / 2
    cy = (top + bottom) / 2
    paste_left = int(cx - region_w / 2)
    paste_top = int(cy - region_h / 2)

    src_x = max(0, -paste_left)
    src_y = max(0, -paste_top)
    dst_x = max(0, paste_left)
    dst_y = max(0, paste_top)
    visible_w = min(region_w - src_x, orig_w - dst_x)
    visible_h = min(region_h - src_y, orig_h - dst_y)

    if visible_w <= 0 or visible_h <= 0:
        return _encode(original)

    region_clip = region.crop((src_x, src_y, src_x + visible_w, src_y + visible_h))
    result = original.copy()
    result.paste(region_clip, (dst_x, dst_y))
    return _encode(result)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
