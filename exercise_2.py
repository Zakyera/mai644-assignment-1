from pathlib import Path

import cv2
import numpy as np


# Part (a): Colour-based fruit segmentation
def segment_fruit(image, ranges):
    """Combine HSV ranges to make a binary fruit mask."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for lower, upper in ranges:
        selected = cv2.inRange(hsv, lower, upper)
        mask = cv2.bitwise_or(mask, selected)

    return mask


def run_part_a():
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "outputs" / "exercise_2a"
    output_dir.mkdir(parents=True, exist_ok=True)

    # HSV separates colour from brightness. OpenCV uses H: 0-179, S and V: 0-255.
    # These ranges were chosen by comparing masks for the supplied images.
    thresholds = {
        # Red crosses the ends of the hue scale, so it needs two ranges.
        "apples": [
            ((0, 45, 25), (15, 255, 255)),
            ((160, 45, 25), (179, 255, 255)),
        ],
        # Include yellow and green apples as well as red ones.
        "apples2": [
            ((0, 40, 45), (85, 255, 255)),
            ((160, 40, 45), (179, 255, 255)),
        ],
        # Blue reflections and dark, weakly saturated parts of the berries.
        "blackberries": [
            ((80, 0, 0), (150, 255, 255)),
            ((0, 0, 0), (179, 120, 95)),
        ],
        # Detect ripe red cherries; green cherries will be missed.
        "cherries": [
            ((0, 80, 35), (12, 255, 255)),
            ((165, 80, 35), (179, 255, 255)),
        ],
    }

    for name, ranges in thresholds.items():
        image_path = project_dir / "assignment-1-data" / f"{name}.jpg"
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read {image_path}")

        mask = segment_fruit(image, ranges)
        segmented = cv2.bitwise_and(image, image, mask=mask)

        for suffix, result in [("mask", mask), ("segmented", segmented)]:
            output_path = output_dir / f"{name}_{suffix}.png"
            if not cv2.imwrite(str(output_path), result):
                raise OSError(f"Could not save {output_path}")
        print(f"Saved mask and segmented image for {name}")


# Part (b): Edge detection, noise and denoising
def add_gaussian_noise(image, psnr):
    """Choose the noise standard deviation using the assignment's formula."""
    intensity_range = float(image.max()) - float(image.min())
    sigma = intensity_range / (10 ** (psnr / 20))
    noise = np.random.normal(0, sigma, image.shape)

    # Add in floating point to avoid uint8 overflow, then prepare for Canny.
    noisy = image.astype(float) + noise
    noisy = np.clip(np.rint(noisy), 0, 255).astype(np.uint8)
    return noisy, sigma


def detect_edges_and_regions(image):
    # Floating point keeps both positive and negative Sobel derivatives.
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Keep the same thresholds for all images so noise effects are comparable.
    _, sobel_edges = cv2.threshold(magnitude, 100, 255, cv2.THRESH_BINARY)
    canny_edges = cv2.Canny(image, 50, 150, L2gradient=True)
    _, dark_regions = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV)
    mid_regions = cv2.inRange(image, 101, 180)

    # Only clip the magnitude for display; threshold the original float values.
    sobel_display = np.clip(np.rint(magnitude), 0, 255).astype(np.uint8)
    return {
        "input": image,
        "sobel_magnitude": sobel_display,
        "sobel_edges": sobel_edges.astype(np.uint8),
        "canny": canny_edges,
        "dark_regions": dark_regions,
        "mid_regions": mid_regions,
    }


def save_results(image, name, output_dir):
    results = detect_edges_and_regions(image)
    for suffix, result in results.items():
        output_path = output_dir / f"{name}_{suffix}.png"
        if not cv2.imwrite(str(output_path), result):
            raise OSError(f"Could not save {output_path}")
    print(f"Saved results for {name}")


def run_part_b():
    project_dir = Path(__file__).resolve().parent
    image_path = project_dir / "assignment-1-data" / "blox.jpg"
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read {image_path}")

    output_dir = project_dir / "outputs" / "exercise_2b"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_results(image, "clean", output_dir)

    # Use the clean image's intensity range to set the noise level.
    intensity_range = float(image.max()) - float(image.min())
    np.random.seed(0)
    for psnr in (20, 10):
        noisy, sigma = add_gaussian_noise(image, psnr)
        name = f"noise_{psnr}db"
        save_results(noisy, name, output_dir)

        # Clipping to 0-255 changes the actual error, especially at 10 dB.
        error = noisy.astype(float) - image.astype(float)
        measured_psnr = 20 * np.log10(intensity_range / np.sqrt(np.mean(error ** 2)))
        print(f"Target: {psnr} dB, sigma: {sigma:.2f}, after clipping: {measured_psnr:.2f} dB")

        # Compare Gaussian and box smoothing, as covered in lecture 3.
        denoised_images = {
            "gaussian_3x3_sigma1": cv2.GaussianBlur(noisy, (3, 3), 1),
            "gaussian_7x7_sigma2": cv2.GaussianBlur(noisy, (7, 7), 2),
            "box_3x3": cv2.blur(noisy, (3, 3)),
            "box_7x7": cv2.blur(noisy, (7, 7)),
        }
        for method, denoised in denoised_images.items():
            save_results(denoised, f"{name}_{method}", output_dir)


def main():
    run_part_a()
    run_part_b()


if __name__ == "__main__":
    main()
