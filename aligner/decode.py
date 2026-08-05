"""Decode photodiode pulse markers + a frame clock into frame-exact stimulus events."""
from __future__ import annotations

import numpy as np

# Pulse count -> block type. A black rest screen carries no marker (0 pulses).
PULSE_TO_TYPE = {1: "grey", 2: "still", 3: "moving"}
TYPE_TO_PULSE = {v: k for k, v in PULSE_TO_TYPE.items()}

# Marker-pulse timing (must match the runner's generator): pulses sit at a fixed period of
# ~0.118 s (6 frames at the 51 fps reference). The default separation threshold sits in the
# stable valley between one pulse's flicker width (< ~50 ms) and that ~118 ms period.
PULSE_PERIOD_S = 6 / 51.0
DEFAULT_PULSE_SEP_S = 0.07
DEFAULT_BURST_GAP_S = 0.8
DEFAULT_MIN_OFFSET = 0.5   # a marker peaks ~1 V above baseline; grating flicker stays < ~0.1 V


def detect_frame_clock(frame_chan, sr, *, hi_thresh=None):
    """Rising-edge time (s) of every frame-clock pulse → the per-frame timestamps.

    ``frame_chan`` is the frame-sync channel, ``sr`` its sample rate (Hz). Returns a 1-D
    array of onset times; index i is the onset of imaging frame i.
    """
    f = np.asarray(frame_chan, dtype=float)
    if hi_thresh is None:
        hi_thresh = (np.nanmax(f) + np.nanmin(f)) / 2.0
    ab = f > hi_thresh
    rising = np.where((~ab[:-1]) & (ab[1:]))[0] + 1
    return rising / float(sr)


def time_to_frame(t, frame_onsets):
    """Index of the frame whose onset is at or just before time ``t`` (nearest at the edges)."""
    frame_onsets = np.asarray(frame_onsets, dtype=float)
    if frame_onsets.size == 0:
        return -1
    i = int(np.searchsorted(frame_onsets, t, side="right") - 1)
    return max(0, min(i, frame_onsets.size - 1))


def detect_pulse_bursts(pulse_chan, sr, t0=0.0, t1=None, *, base_window_s=2.0,
                        min_offset=DEFAULT_MIN_OFFSET, burst_gap_s=DEFAULT_BURST_GAP_S,
                        pulse_sep_s=DEFAULT_PULSE_SEP_S):
    """Count marker pulses within ``[t0, t1]`` s, grouped into per-block bursts.

    Returns ``[(onset_s, n_pulses), ...]`` where ``n_pulses`` in {1, 2, 3} is the block type
    per :data:`PULSE_TO_TYPE`, and ``onset_s`` is the first pulse's onset (s, absolute).

    A photodiode marker is not a clean step: each logical pulse is a short burst of ~120 Hz
    monitor flicker spanning ~0.3–1.4 V, so one pulse crosses any fixed threshold many times.
    Counting raw threshold crossings over-counts. Instead the supra-threshold run onsets are
    grouped into *pulses* by onset-to-onset spacing measured from the current pulse's first
    onset: a run more than ``pulse_sep_s`` from that onset starts a new pulse, otherwise it is
    the same pulse's flicker. Keying on the stable inter-pulse period (not the width-dependent
    end-to-start gap) keeps the count robust to acquisition lag. A grating block adds only
    low-amplitude flicker (< ~0.1 V) that never reaches ``min_offset``, so it is ignored.
    """
    p = np.asarray(pulse_chan, dtype=float)
    n = p.size
    t1 = n / float(sr) if t1 is None else t1
    a, b = int(t0 * sr), int(t1 * sr)
    base_win = p[a:int((t0 + base_window_s) * sr)]
    base = float(np.nanmedian(base_win)) if base_win.size else 0.0
    seg = p[a:b]
    if not np.isfinite(base):
        base = float(np.nanmin(seg)) if np.isfinite(seg).any() else 0.0

    ab = np.where(np.isfinite(seg), seg, -np.inf) > (base + min_offset)
    if not ab.any():
        return []

    # contiguous supra-threshold runs → onset sample indices
    d = np.diff(ab.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    if ab[0]:
        starts = [0] + starts
    onset_s = np.asarray(starts, dtype=float) / sr + t0

    # group flicker sub-runs into marker pulses by onset-to-onset spacing
    pulses = [onset_s[0]]
    for o in onset_s[1:]:
        if o - pulses[-1] > pulse_sep_s:
            pulses.append(o)
    pulses = np.asarray(pulses)

    # group pulses into per-block bursts by the larger inter-block gap
    bursts, cur = [], [pulses[0]]
    for r in pulses[1:]:
        if r - cur[-1] > burst_gap_s:
            bursts.append((float(cur[0]), len(cur)))
            cur = [r]
        else:
            cur.append(r)
    bursts.append((float(cur[0]), len(cur)))
    return bursts


def decode_recording(frame_chan, pulse_chan, sr, **kwargs):
    """Decode a whole recording into frame-exact marker events.

    Returns ``[{type, n_pulses, onset_s, onset_frame}, ...]`` in time order, one per marker
    burst (grey/still/moving). Black rest screens carry no marker and do not appear.
    Unknown pulse counts (e.g. 4+) are typed ``"?<n>"`` rather than dropped, so a mis-decode
    is visible instead of silent.
    """
    frame_onsets = detect_frame_clock(frame_chan, sr)
    events = []
    for onset_s, n_pulses in detect_pulse_bursts(pulse_chan, sr, **kwargs):
        events.append({
            "type": PULSE_TO_TYPE.get(n_pulses, f"?{n_pulses}"),
            "n_pulses": int(n_pulses),
            "onset_s": float(onset_s),
            "onset_frame": time_to_frame(onset_s, frame_onsets),
        })
    return events
