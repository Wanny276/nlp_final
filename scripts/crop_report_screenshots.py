"""Create focused report crops from the full Streamlit screenshots."""

from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "report" / "figures"

CROPS = {
    "ui-single.png": ("ui-single-focus.png", (260, 80, 1420, 1000)),
    "ui-batch.png": ("ui-batch-focus.png", (260, 80, 1420, 1000)),
    "ui-test.png": ("ui-test-focus.png", (260, 80, 1420, 930)),
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
