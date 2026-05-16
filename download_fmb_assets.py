"""Download FMB (Functional Manipulation Benchmark) STEP files."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import requests
import tyro
from tqdm import tqdm

BASE_URL = "https://functional-manipulation-benchmark.github.io/static/files"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "assets" / "urdf" / "fmb" / "raw_step"

FMB_FILES: Dict[str, str] = {
    "peg.step": f"{BASE_URL}/peg.step",
    "peg_board.step": f"{BASE_URL}/peg_board.step",
    "peg_fixture.step": f"{BASE_URL}/peg%20fixture.step",
    "board_1.step": f"{BASE_URL}/Board%201.step",
    "board_2.step": f"{BASE_URL}/Board%202.step",
    "board_3.step": f"{BASE_URL}/Board%203.step",
    "board_fixture.step": f"{BASE_URL}/board%20fixture.step",
}


def download_file(url: str, output_path: Path, skip_existing: bool = True) -> None:
    if skip_existing and output_path.exists():
        print(f"  Skipping {output_path.name} (already exists)")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with output_path.open("wb") as f, tqdm(
        desc=f"  {output_path.name}",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            bar.update(size)


@dataclass
class DownloadArgs:
    """Download FMB STEP files from the official website."""

    output_dir: Path = DEFAULT_OUTPUT_DIR
    skip_existing: bool = True
    """Skip files that already exist locally."""
    file: Optional[str] = None
    """Download a single file by local name (e.g. 'peg.step'). Downloads all if omitted."""
    list: bool = False
    """Print available files and exit."""


def main() -> None:
    args: DownloadArgs = tyro.cli(DownloadArgs)

    if args.list:
        print("Available FMB STEP files:")
        for local_name, url in FMB_FILES.items():
            print(f"  {local_name:25s} <- {url}")
        return

    files = FMB_FILES
    if args.file is not None:
        if args.file not in FMB_FILES:
            raise ValueError(
                f"Unknown file '{args.file}'. Choose from: {list(FMB_FILES.keys())}"
            )
        files = {args.file: FMB_FILES[args.file]}

    print(f"Downloading {len(files)} FMB STEP file(s) to {args.output_dir}/\n")
    for local_name, url in files.items():
        download_file(url, args.output_dir / local_name, skip_existing=args.skip_existing)

    print("\nDone!")


if __name__ == "__main__":
    main()
