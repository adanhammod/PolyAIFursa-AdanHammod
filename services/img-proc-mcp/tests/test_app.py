import base64
import io

import pytest
from PIL import Image

from app import _decode, _encode, rotate, flip, blur, resize, crop, add_noise, replace_region


@pytest.fixture
def sample_b64():
    img = Image.new("RGB", (100, 80), color=(128, 64, 32))
    return _encode(img)


def assert_valid_png(b64_str: str) -> None:
    data = base64.b64decode(b64_str)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_rotate(sample_b64):
    assert_valid_png(rotate(sample_b64, 45.0))


def test_flip_horizontal(sample_b64):
    assert_valid_png(flip(sample_b64, "horizontal"))


def test_flip_vertical(sample_b64):
    assert_valid_png(flip(sample_b64, "vertical"))


def test_flip_invalid_direction(sample_b64):
    with pytest.raises(ValueError):
        flip(sample_b64, "diagonal")


def test_blur(sample_b64):
    assert_valid_png(blur(sample_b64, radius=3.0))


def test_resize_changes_dimensions(sample_b64):
    result = resize(sample_b64, 50, 40)
    assert _decode(result).size == (50, 40)


def test_resize_invalid_width(sample_b64):
    with pytest.raises(ValueError):
        resize(sample_b64, 0, 40)


def test_resize_invalid_height(sample_b64):
    with pytest.raises(ValueError):
        resize(sample_b64, 50, -1)


def test_crop_changes_dimensions(sample_b64):
    result = crop(sample_b64, 10, 5, 60, 55)
    assert _decode(result).size == (50, 50)


def test_crop_out_of_bounds(sample_b64):
    with pytest.raises(ValueError):
        crop(sample_b64, 0, 0, 200, 200)


def test_add_noise(sample_b64):
    assert_valid_png(add_noise(sample_b64, amount=0.05))


def test_add_noise_invalid_high(sample_b64):
    with pytest.raises(ValueError):
        add_noise(sample_b64, amount=1.5)


def test_add_noise_invalid_negative(sample_b64):
    with pytest.raises(ValueError):
        add_noise(sample_b64, amount=-0.1)


def test_replace_region_same_size(sample_b64):
    # Patch a 20x20 region at box (10, 10, 30, 30); region same size as box → exact placement
    region_img = Image.new("RGB", (20, 20), color=(255, 0, 0))
    region_b64 = _encode(region_img)
    result_b64 = replace_region(sample_b64, region_b64, left=10, top=10, right=30, bottom=30)
    assert_valid_png(result_b64)
    result = _decode(result_b64)
    assert result.size == (100, 80)
    assert result.getpixel((20, 20)) == (255, 0, 0)  # center of pasted region


def test_replace_region_oversized_clips(sample_b64):
    # Processed region larger than box (e.g. rotated) — must clip, not raise
    large_region = Image.new("RGB", (60, 60), color=(0, 255, 0))
    region_b64 = _encode(large_region)
    result_b64 = replace_region(sample_b64, region_b64, left=80, top=60, right=90, bottom=70)
    assert_valid_png(result_b64)
    result = _decode(result_b64)
    assert result.size == (100, 80)  # original size preserved


def test_replace_region_fully_outside_returns_original(sample_b64):
    # Region centered so far outside image bounds that nothing is pasted
    tiny_region = Image.new("RGB", (2, 2), color=(0, 0, 255))
    region_b64 = _encode(tiny_region)
    result_b64 = replace_region(sample_b64, region_b64, left=200, top=200, right=210, bottom=210)
    assert_valid_png(result_b64)
    result = _decode(result_b64)
    assert result.size == (100, 80)
