"""Create focused report crops from the full Streamlit screenshots."""

from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "report" / "figures"

CROPS = {
    "pages/single-zh.png": ("ui-single-focus.png", (0, 0, 1873, 1660)),
    "pages/batch.png": ("ui-batch-focus.png", (0, 1580, 1873, 3920)),
    "pages/test.png": ("ui-test-focus.png", (0, 1180, 1873, 2760)),
    "pages/evaluate.png": ("ui-evaluate-focus.png", (0, 0, 1873, 2260)),
}


def main() -> None:
    for source_name, (output_name, box) in CROPS.items():
        source = FIGURE_DIR / source_name
        output = FIGURE_DIR / output_name
        with Image.open(source) as image:
            image.crop(box).save(output, optimize=True)
        print(f"{source_name} -> {output_name}")


if __name__ == "__main__":
    main()
