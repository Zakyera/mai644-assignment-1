from math import floor
from pathlib import Path

import cv2
import numpy as np


def nn_resize(image, w, h):
    """Resize an image using nn interpolation."""
    if w <= 0 or h <= 0:
        raise ValueError("Width and height must be positive.")

    old_h, old_w = image.shape[:2]
    resized = np.zeros((h, w) + image.shape[2:], dtype=image.dtype)
    scale_x = old_w / w
    scale_y = old_h / h

    # we map the pixel centres back to the input,and then round to the nearest pixel.
    # floor(value + 0.5) will choose the higher index when two pixels are equally close.
    for y in range(h):
        source_y = (y + 0.5) * scale_y - 0.5
        nearest_y = min(max(floor(source_y + 0.5), 0), old_h - 1)

        for x in range(w):
            source_x = (x + 0.5) * scale_x - 0.5
            nearest_x = min(max(floor(source_x + 0.5), 0), old_w - 1)
            resized[y, x] = image[nearest_y, nearest_x]

    return resized


def main():
    project_dir = Path(__file__).resolve().parent
    image_path = project_dir / "assignment-1-data" / "goldfish.jpg"
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read {image_path}")

    height, width = image.shape[:2]
    output_dir = project_dir / "outputs" / "exercise_1a"
    output_dir.mkdir(parents=True, exist_ok=True)

    sizes = [
        ("upscaled_2x", width * 2, height * 2),
        ("downscaled_half", width // 2, height // 2),
    ]

    print(f"Original image: {width} x {height}")
    for name, new_w, new_h in sizes:
        resized = nn_resize(image, new_w, new_h)
        output_path = output_dir / f"goldfish_{name}.png"
        if not cv2.imwrite(str(output_path), resized):
            raise OSError(f"Could not save {output_path}")
        print(f"Saved {new_w} x {new_h}: {output_path}")


if __name__ == "__main__":
    main()
