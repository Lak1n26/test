#!/usr/bin/env python3
"""
Script for calculating WER/CER metrics between predictions and ground truth.

Usage:
    python calc_metrics.py --predictions /path/to/predictions --ground_truth /path/to/transcriptions

Both directories should contain .txt files with the same names.
"""

import argparse
from pathlib import Path

from src.metrics.utils import calc_cer, calc_wer, calc_wer_detailed


def load_texts_from_dir(directory: Path) -> dict:
    """
    Load all text files from a directory.

    Args:
        directory (Path): path to directory with .txt files.
    Returns:
        texts (dict): mapping from filename (without extension) to text content.
    """
    texts = {}
    for txt_file in sorted(directory.glob("*.txt")):
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
        texts[txt_file.stem] = text
    return texts


def calculate_metrics(predictions: dict, ground_truth: dict, verbose: bool = False):
    """
    Calculate WER and CER metrics.

    Args:
        predictions (dict): mapping from utterance_id to predicted text.
        ground_truth (dict): mapping from utterance_id to ground truth text.
        verbose (bool): if True, print per-utterance metrics.
    Returns:
        results (dict): aggregated metrics.
    """
    total_wer = 0.0
    total_cer = 0.0
    total_words = 0
    total_chars = 0
    total_word_errors = 0
    total_char_errors = 0

    matched = 0
    missing_gt = []
    missing_pred = []

    for utt_id in sorted(set(predictions.keys()) | set(ground_truth.keys())):
        if utt_id not in predictions:
            missing_pred.append(utt_id)
            continue
        if utt_id not in ground_truth:
            missing_gt.append(utt_id)
            continue

        pred = predictions[utt_id]
        gt = ground_truth[utt_id]

        wer = calc_wer(gt, pred)
        cer = calc_cer(gt, pred)

        # Count for corpus-level metrics
        gt_words = gt.strip().lower().split()
        gt_chars = list(gt.strip().lower())

        # Get detailed metrics for corpus-level calculation
        wer_detail = calc_wer_detailed(gt, pred)

        total_words += len(gt_words)
        total_chars += len(gt_chars)
        total_word_errors += wer_detail["substitutions"] + wer_detail["deletions"] + wer_detail["insertions"]

        # For CER, calculate edit distance
        from src.metrics.utils import levenshtein_distance
        pred_chars = list(pred.strip().lower())
        char_errors = levenshtein_distance(gt_chars, pred_chars)
        total_char_errors += char_errors

        total_wer += wer
        total_cer += cer
        matched += 1

        if verbose:
            print(f"\n[{utt_id}]")
            print(f"  GT:   {gt}")
            print(f"  Pred: {pred}")
            print(f"  WER: {wer*100:.2f}%, CER: {cer*100:.2f}%")

    # Calculate averages
    avg_wer = total_wer / matched if matched > 0 else 0
    avg_cer = total_cer / matched if matched > 0 else 0

    # Calculate corpus-level metrics
    corpus_wer = total_word_errors / total_words if total_words > 0 else 0
    corpus_cer = total_char_errors / total_chars if total_chars > 0 else 0

    results = {
        "matched_utterances": matched,
        "missing_predictions": len(missing_pred),
        "missing_ground_truth": len(missing_gt),
        "avg_wer": avg_wer,
        "avg_cer": avg_cer,
        "corpus_wer": corpus_wer,
        "corpus_cer": corpus_cer,
        "total_words": total_words,
        "total_word_errors": total_word_errors,
        "total_chars": total_chars,
        "total_char_errors": total_char_errors,
    }

    return results, missing_pred, missing_gt


def main():
    parser = argparse.ArgumentParser(
        description="Calculate WER/CER metrics between predictions and ground truth."
    )
    parser.add_argument(
        "--predictions", "-p",
        type=str,
        required=True,
        help="Path to directory with prediction .txt files",
    )
    parser.add_argument(
        "--ground_truth", "-g",
        type=str,
        required=True,
        help="Path to directory with ground truth .txt files",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-utterance metrics",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save results as JSON",
    )

    args = parser.parse_args()

    pred_dir = Path(args.predictions)
    gt_dir = Path(args.ground_truth)

    if not pred_dir.exists():
        raise ValueError(f"Predictions directory does not exist: {pred_dir}")
    if not gt_dir.exists():
        raise ValueError(f"Ground truth directory does not exist: {gt_dir}")

    print(f"Loading predictions from: {pred_dir}")
    print(f"Loading ground truth from: {gt_dir}")

    predictions = load_texts_from_dir(pred_dir)
    ground_truth = load_texts_from_dir(gt_dir)

    print(f"\nFound {len(predictions)} predictions")
    print(f"Found {len(ground_truth)} ground truth files")

    results, missing_pred, missing_gt = calculate_metrics(
        predictions, ground_truth, verbose=args.verbose
    )

    # Print results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Matched utterances: {results['matched_utterances']}")
    print(f"Missing predictions: {results['missing_predictions']}")
    print(f"Missing ground truth: {results['missing_ground_truth']}")
    print()
    print(f"Average WER: {results['avg_wer']*100:.2f}%")
    print(f"Average CER: {results['avg_cer']*100:.2f}%")
    print()
    print(f"Corpus WER: {results['corpus_wer']*100:.2f}% ({results['total_word_errors']}/{results['total_words']} words)")
    print(f"Corpus CER: {results['corpus_cer']*100:.2f}% ({results['total_char_errors']}/{results['total_chars']} chars)")
    print("=" * 50)

    # Warn about missing files
    if missing_pred:
        print(f"\nWarning: {len(missing_pred)} ground truth files have no predictions")
        if args.verbose:
            for utt_id in missing_pred[:10]:
                print(f"  - {utt_id}")
            if len(missing_pred) > 10:
                print(f"  ... and {len(missing_pred) - 10} more")

    if missing_gt:
        print(f"\nWarning: {len(missing_gt)} predictions have no ground truth")
        if args.verbose:
            for utt_id in missing_gt[:10]:
                print(f"  - {utt_id}")
            if len(missing_gt) > 10:
                print(f"  ... and {len(missing_gt) - 10} more")

    # Save results if requested
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
