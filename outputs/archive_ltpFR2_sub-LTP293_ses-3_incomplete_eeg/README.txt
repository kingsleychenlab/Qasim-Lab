ARCHIVED — INCOMPLETE EEG. DO NOT USE AS CANONICAL.

Session: sub-LTP293 / ses-3 / task-ltpFR2
Reason : EEG recording (EDF) is TRUNCATED relative to the behavioral session.
         The EDF is only 2177 s, but word presentations extend to ~3291 s.
         With a 300-800 ms extraction window, only 449/576 words yield a
         window that fits inside the recording; 127 windows fall off the end.
         First out-of-bounds word: trial 19, serialpos 18, COLLEGE.

A session is only valid for canonical EEG extraction if ALL 576 WORD events
have a full [300, 800] ms window inside raw.n_times (checked via MNE).

Superseded on 2026-07-04 by a session with 576/576 valid EEG windows.
