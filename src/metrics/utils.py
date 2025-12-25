"""
Utility functions for calculating ASR metrics.
"""


def levenshtein_distance(ref: list, hyp: list) -> int:
    """
    Calculate Levenshtein (edit) distance between two sequences.

    Uses dynamic programming with O(n*m) time and O(min(n,m)) space.

    Args:
        ref (list): Reference sequence.
        hyp (list): Hypothesis sequence.
    Returns:
        distance (int): Levenshtein distance.
    """
    n, m = len(ref), len(hyp)

    # Optimize for edge cases
    if n == 0:
        return m
    if m == 0:
        return n

    # Use shorter sequence for columns (space optimization)
    if n < m:
        ref, hyp = hyp, ref
        n, m = m, n

    # Initialize previous and current row
    prev_row = list(range(m + 1))
    curr_row = [0] * (m + 1)

    for i in range(1, n + 1):
        curr_row[0] = i
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                curr_row[j] = prev_row[j - 1]
            else:
                curr_row[j] = 1 + min(
                    prev_row[j],      # deletion
                    curr_row[j - 1],  # insertion
                    prev_row[j - 1],  # substitution
                )
        prev_row, curr_row = curr_row, prev_row

    return prev_row[m]


def calc_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER).

    WER = edit_distance(ref_words, hyp_words) / len(ref_words)

    Args:
        reference (str): Ground truth text.
        hypothesis (str): Predicted text.
    Returns:
        wer (float): Word Error Rate (0 to infinity, but typically 0-1).
    """
    # Split into words
    ref_words = reference.strip().lower().split()
    hyp_words = hypothesis.strip().lower().split()

    # Handle empty reference
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else float("inf")

    # Calculate edit distance
    distance = levenshtein_distance(ref_words, hyp_words)

    return distance / len(ref_words)


def calc_cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate (CER).

    CER = edit_distance(ref_chars, hyp_chars) / len(ref_chars)

    Args:
        reference (str): Ground truth text.
        hypothesis (str): Predicted text.
    Returns:
        cer (float): Character Error Rate (0 to infinity, but typically 0-1).
    """
    # Convert to character lists
    ref_chars = list(reference.strip().lower())
    hyp_chars = list(hypothesis.strip().lower())

    # Handle empty reference
    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else float("inf")

    # Calculate edit distance
    distance = levenshtein_distance(ref_chars, hyp_chars)

    return distance / len(ref_chars)


def calc_wer_detailed(reference: str, hypothesis: str) -> dict:
    """
    Calculate WER with detailed breakdown.

    Returns substitutions, deletions, insertions counts.

    Args:
        reference (str): Ground truth text.
        hypothesis (str): Predicted text.
    Returns:
        result (dict): Dictionary with wer, substitutions, deletions,
            insertions, and reference length.
    """
    ref_words = reference.strip().lower().split()
    hyp_words = hypothesis.strip().lower().split()

    n, m = len(ref_words), len(hyp_words)

    if n == 0:
        return {
            "wer": 0.0 if m == 0 else float("inf"),
            "substitutions": 0,
            "deletions": 0,
            "insertions": m,
            "ref_length": 0,
        }

    # DP table for edit distance with backtracking
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    # Backtrack to count operations
    i, j = n, m
    substitutions = deletions = insertions = 0

    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1

    return {
        "wer": dp[n][m] / n,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "ref_length": n,
    }

