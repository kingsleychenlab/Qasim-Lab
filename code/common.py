#!/usr/bin/env python3
"""
Shared helpers.

Every stage in this pipeline writes a human-readable report while it runs and
counts its own PASS/FAIL checks, and several need the PEERS word pool. Most
scripts had reimplemented each of those, so they live here now as one copy.

Imported as a plain sibling module (`from common import Tee`). That works
because a script's own directory is on sys.path when it is run directly, and
because step09/step10 already add code/ to sys.path before importing siblings.
"""

import pandas as pd

BAR = "=" * 78


class Tee:
    """Print to stdout and mirror every line into a report file.

    Also tallies failed checks and warnings, so a script can end with
    `sys.exit(1 if log.fail else 0)` instead of threading a counter through by
    hand. Call the instance to write a line:

        log = Tee(open(path, "w"))
        log("some text")
        log.check("576 trials present", len(df) == 576, f"got {len(df)}")
        sys.exit(1 if log.fail else 0)
    """

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
        """Record and print one PASS/FAIL check. Returns ok, so it can be used
        inline in an `if`."""
        if not ok:
            self.fail += 1
        self(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
        return bool(ok)

    def warn(self, label, detail=""):
        """Flag something suspect that is not fatal. Counted separately from
        failures so a script can report both without conflating them."""
        self.warnings += 1
        self(f"[WARN] {label}" + (f" -- {detail}" if detail else ""))


def peers_word_set(path):
    """The 576-word PEERS pool, uppercased, for membership tests.

    Uppercased because the events files and the word pool disagree on casing.
    """
    return set(pd.read_csv(path).word.str.upper())


def load_word_to_row(path):
    """Map word (uppercased) -> its row index in the T5 embedding matrix.

    The lookup that aligns a trial's word with its target vector. Raises if the
    pool has duplicate words after case-folding, since that would silently give
    two different words the same embedding row.
    """
    order = pd.read_csv(path)
    upper = order.word.str.upper()
    if upper.duplicated().any():
        dups = upper[upper.duplicated()].tolist()
        raise ValueError(f"duplicate words in {path}: {dups[:10]}")
    return dict(zip(upper, order.row_index.astype(int)))
