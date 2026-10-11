from pathlib import Path

import cv2
import numpy as np


def make_box_filter(s):
    """we create an s by s filter whose values add up to one."""
    if s < 1:
        raise ValueError("Filter size must be positive.")

    box = np.ones((s, s), dtype=float)
    return box / box.sum()


def convolve_image(image, filter):
    """we apply a two-dimensional filter to each image channel."""
    size = filter.shape[0]
    pad_before = (size - 1) // 2
    pad_after = size // 2

    grayscale = image.ndim == 2
    if grayscale:
        image = image[:, :, np.newaxis]

    height, width, channels = image.shape
    padded = np.pad(
        image,
        ((pad_before, pad_after), (pad_before, pad_after), (0, 0)),
        mode="edge",
    )
    result = np.zeros_like(image)
    flat_filter = filter[::-1, ::-1].flatten()

    for y in range(height):
        for x in range(width):
            for channel in range(channels):
                window = padded[y:y + size, x:x + size, channel]
                value = np.dot(window.flatten(), flat_filter)
                result[y, x, channel] = np.clip(round(value), 0, 255)

    if grayscale:
        return result[:, :, 0]
    return result


def main():
    project_dir = Path(__file__).resolve().parent
    image_path = project_dir / "assignment-1-data" / "goldfish.jpg"
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read {image_path}")

    output_dir = project_dir / "outputs" / "exercise_1b"
    output_dir.mkdir(parents=True, exist_ok=True)

    for size in (3, 9):
        box = make_box_filter(size)
        blurred = convolve_image(image, box)
        output_path = output_dir / f"goldfish_box_{size}x{size}.png"
        if not cv2.imwrite(str(output_path), blurred):
            raise OSError(f"Could not save {output_path}")
        print(f"Saved {size} x {size} blur: {output_path}")


if __name__ == "__main__":
    main()
