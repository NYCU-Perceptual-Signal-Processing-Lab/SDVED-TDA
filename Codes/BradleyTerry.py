"""Convert pairwise comparisons into Bradley-Terry scores."""

import argparse
import csv
import re
from pathlib import Path

import choix
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_CSV = SCRIPT_DIR / "outputs" / "bradley_terry_scores.csv"


def parse_comparisons(path):
    """Parse `attribute: winner loser ...` lines into integer ID pairs."""
    comparisons = []
    name_to_id = {}
    id_to_name = []

    def get_item_id(name):
        if name not in name_to_id:
            name_to_id[name] = len(id_to_name)
            id_to_name.append(name)
        return name_to_id[name]

    with path.open("r", encoding="utf-8") as input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                raise ValueError(
                    f"Line {line_number} has no ':'. Expected "
                    "'attribute: winner loser [winner loser ...]'."
                )

            _attribute, pair_text = line.split(":", 1)
            items = [token for token in re.split(r"[\s,]+", pair_text.strip()) if token]
            if len(items) % 2:
                raise ValueError(
                    f"Line {line_number} contains an odd number of item names."
                )

            for index in range(0, len(items), 2):
                winner_id = get_item_id(items[index])
                loser_id = get_item_id(items[index + 1])
                comparisons.append((winner_id, loser_id))

    return comparisons, id_to_name


def calculate_scores(comparisons, item_names, alpha):
    if not comparisons or not item_names:
        raise ValueError("No pairwise comparisons were found in the input file.")

    params = choix.ilsr_pairwise(
        n_items=len(item_names), data=comparisons, alpha=alpha
    )
    minimum = np.min(params)
    maximum = np.max(params)
    centered = params - np.mean(params)
    standard_deviation = np.std(params)

    minmax = ((params - minimum) / (maximum - minimum + 1e-9)) * 10
    sigmoid = (1 / (1 + np.exp(-centered))) * 10
    zscore = np.clip(5 + (centered / (standard_deviation + 1e-9)) * 2, 0, 10)

    results = [
        {
            "item": item_names[index],
            "score": float(zscore[index]),
            "raw_score": float(params[index]),
            "minmax_score": float(minmax[index]),
            "sigmoid_score": float(sigmoid[index]),
        }
        for index in range(len(item_names))
    ]
    return sorted(results, key=lambda item: item["raw_score"], reverse=True)


def write_results(results, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit a Bradley-Terry model to winner/loser comparisons."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Comparison text file; see README.txt for the expected format.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_CSV),
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_CSV}).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="L2 regularization used by choix (default: 1.0).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_path.is_file():
        raise SystemExit(f"Input file does not exist: {input_path}")
    if args.alpha < 0:
        raise SystemExit("--alpha must be non-negative.")

    try:
        comparisons, item_names = parse_comparisons(input_path)
        results = calculate_scores(comparisons, item_names, args.alpha)
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"Could not calculate scores: {exc}") from exc

    write_results(results, output_path)
    print(f"Parsed {len(comparisons)} comparisons for {len(item_names)} items.")
    print(f"Saved scores to {output_path}")


if __name__ == "__main__":
    main()
