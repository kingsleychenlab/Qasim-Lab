#!/usr/bin/env python3
"""Shared helpers: the report writer/check-counter (Tee) and the PEERS word pool.

Imported as a plain sibling module (``from common import Tee``); a script's own
directory is on sys.path when run directly.
"""

import pandas as pd

BAR = "=" * 78


class Tee:
    """Print to stdout, mirror every line into a report file, and tally failed
    checks and warnings so a script can end with ``sys.exit(1 if log.fail else 0)``."""

    def __init__(self, fh):
        self.fh = fh
        self.fail = 0
        self.warnings = 0

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        # flush: the scaling stages spawn subprocesses that write to the same
        # stdout, and buffered parent output would interleave out of order.
        print(line, flush=True)
        self.fh.write(line + "\n")

    def rule(self, title=""):
        """A section divider, optionally boxing a title."""
        self("\n" + BAR)
        if title:
            self(title)
            self(BAR)

    def check(self, label, ok, detail=""):
        """Record and print one PASS/FAIL check; returns ok for inline use in an `if`."""
        if not ok:
            self.fail += 1
        self(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
        return bool(ok)

    def warn(self, label, detail=""):
        """Flag something suspect but not fatal; counted separately from failures."""
        self.warnings += 1
        self(f"[WARN] {label}" + (f" -- {detail}" if detail else ""))


def peers_word_set(path):
    """The 576-word PEERS pool, uppercased (events and pool disagree on casing)."""
    return set(pd.read_csv(path).word.str.upper())


def load_word_to_row(path):
    """Map word (uppercased) -> its row index in the T5 embedding matrix.

    Raises on duplicate words after case-folding, which would give two words the
    same embedding row.
    """
    pool = pd.read_csv(path)
    upper_words = pool.word.str.upper()
    if upper_words.duplicated().any():
        duplicates = upper_words[upper_words.duplicated()].tolist()
        raise ValueError(f"duplicate words in {path}: {duplicates[:10]}")
    return dict(zip(upper_words, pool.row_index.astype(int)))
