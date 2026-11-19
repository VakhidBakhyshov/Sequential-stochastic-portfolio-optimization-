"""Export the supplied Figure 1 artwork to the usual figure output path."""
from pathlib import Path
from shutil import copyfile

ROOT = Path(__file__).resolve().parent


def main():
    source = ROOT / "reference" / "figures" / "fig_pipeline.png"
    copyfile(source, ROOT / "fig_pipeline.png")
    print("saved fig_pipeline.png")


if __name__ == "__main__":
    main()
