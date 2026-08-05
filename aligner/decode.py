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

    If the channel is already HIGH at sample 0 (a recording trimmed to start on a frame),
    that first frame is a rising edge too and is emitted at t=0 — otherwise every frame
    index would shift by one. NaN samples are treated as low (a dropped sample mid-pulse
    must not split one frame into two edges).
    """
    f = np.asarray(frame_chan, dtype=float)
    if hi_thresh is None:
        hi_thresh = (np.nanmax(f) + np.nanmin(f)) / 2.0
    # forward-fill NaN: a dropped sample carries the previous value, so it can't split one
    # frame pulse into two phantom edges (setting NaN low would do exactly that).
    mask = np.isnan(f)
    if mask.any():
        idx = np.where(~mask, np.arange(len(f)), 0)
        np.maximum.accumulate(idx, out=idx)
        f = f[idx]
    ab = f > hi_thresh
    rising = list(np.where((~ab[:-1]) & (ab[1:]))[0] + 1)
    if ab[0]:                       # already high at sample 0 → frame 0 starts at t=0
        rising = [0] + rising
    return np.asarray(rising, dtype=float) / float(sr)


def time_to_frame(t, frame_onsets):
    """Index of the frame whose onset is at or just before time ``t`` (nearest at the edges)."""
    frame_onsets = np.asarray(frame_onsets, dtype=float)
    if frame_onsets.size == 0:
        return -1
    i = int(np.searchsorted(frame_onsets, t, side="right") - 1)
    return max(0, min(i, frame_onsets.size - 1))


def detect_pulse_bursts(pulse_chan, sr, t0=0.0, t1=None, *, base_window_s=2.0,
                        min_offset=None, burst_gap_s=DEFAULT_BURST_GAP_S,
                        pulse_period_s=PULSE_PERIOD_S, base_pct=10.0, span_frac=0.35):
    """Count marker pulses within ``[t0, t1]`` s, grouped into per-block bursts.

    Returns ``[(onset_s, n_pulses), ...]`` where ``n_pulses`` in {1, 2, 3} is the block type
    per :data:`PULSE_TO_TYPE`, and ``onset_s`` is the first pulse's onset (s, absolute).

    A photodiode marker is not a clean step: each logical pulse is a short burst of ~120 Hz
    monitor flicker spanning ~0.3–1.4 V, so one pulse crosses any fixed threshold many times.
    Counting raw threshold crossings over-counts, and a bare onset-to-onset gap threshold
    both *merges* two pulses (if the signal never dips between them — monitor persistence) and
    *splits* one wide pulse (if its own flicker outlasts the gap). Instead each supra-threshold
    run onset is assigned to a marker pulse by the KNOWN, fixed inter-pulse period:
    ``index = round((onset − first_onset) / pulse_period_s)``. The pulse count is the number of
    DISTINCT indices, which is robust to both merges and wide flashes (and to acquisition lag,
    since it keys on the stable period, not a width-dependent gap).

    The threshold is derived from the signal, not hard-coded: ``base`` is a low percentile
    (``base_pct``) of the whole segment (robust to a bright block sitting in a fixed leading
    window), and, unless ``min_offset`` is given, the offset is ``span_frac`` of the
    base-to-peak span (so it scales with photodiode gain). A grating block's low-amplitude
    flicker stays below that and is ignored.
    """
    p = np.asarray(pulse_chan, dtype=float)
    n = p.size
    t1 = n / float(sr) if t1 is None else t1
    a, b = int(t0 * sr), int(t1 * sr)
    seg = np.asarray(p[a:b], dtype=float)
    finite = seg[np.isfinite(seg)]
    if finite.size == 0:
        return []

    # baseline = low percentile of the WHOLE segment (robust to a bright block in a fixed
    # leading window); peak = a HIGH percentile, i.e. the marker level (markers are rare, so
    # a mid percentile would sit on the grating flicker, not the marker). The threshold sits a
    # fraction of the base-to-peak span up, so it scales with photodiode gain and clears the
    # low-amplitude grating flicker. Markers are ~10× brighter than grating flicker, so any
    # span_frac in a wide band works; 0.35 is comfortably between them.
    base = float(np.percentile(finite, base_pct))
    peak = float(np.percentile(finite, 99.5))
    if min_offset is None:
        span = peak - base
        offset = span_frac * span if span > 0 else DEFAULT_MIN_OFFSET
    else:
        offset = float(min_offset)
    thr = base + offset

    ab = np.where(np.isfinite(seg), seg, -np.inf) > thr
    if not ab.any():
        return []

    # contiguous supra-threshold RUNS as (start_s, end_s). We need the END too, not just the
    # onset: two pulses with no dip between them (monitor persistence) form ONE run with no
    # interior edge, so the missing pulse can only be recovered from the run's DURATION.
    d = np.diff(ab.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if ab[0]:
        starts = [0] + starts
    if ab[-1]:
        ends = ends + [len(ab)]
    runs = [(s / sr + t0, e / sr + t0) for s, e in zip(starts, ends)]

    # split runs into per-block bursts by the larger inter-block gap (start-to-previous-start)
    bursts_runs, cur = [], [runs[0]]
    for r in runs[1:]:
        if r[0] - cur[-1][0] > burst_gap_s:
            bursts_runs.append(cur)
            cur = [r]
        else:
            cur.append(r)
    bursts_runs.append(cur)

    # count the DISTINCT pulse slots each burst covers, on the KNOWN fixed period: every run
    # contributes the period indices it spans, from round(start/period) to round(end/period).
    # A single pulse's flicker sub-runs all round to one slot; a merged multi-pulse run spans
    # several slots; a wide flash likewise — all robust to acquisition lag (no width-dependent
    # gap threshold). Assumes the true pulse width < ~half the period (else width≈period is
    # fundamentally indistinguishable from two pulses by threshold crossings alone).
    # A pulse's start sits at k·period and its trailing flicker reaches ~k·period + width,
    # with width ≈ half the period. So a sub-run's time in period units lies in [k, k+~0.5].
    # Assign it to slot ``floor(x + MARGIN)``: MARGIN (period units) must exceed float error
    # at the integer boundary yet stay below 0.5 so the trailing flicker still floors to k.
    # A merged (bridged) run spans several slots; taking the range from its start slot to its
    # END slot counts them. Assumes the true pulse width < ~half the period (width≈period is
    # fundamentally indistinguishable from two pulses by threshold crossings alone).
    MARGIN = 0.1
    bursts = []
    for burst in bursts_runs:
        first = burst[0][0]
        slots = set()
        for (s, e) in burst:
            i0 = int((s - first) / pulse_period_s + MARGIN)
            i1 = int((e - first) / pulse_period_s + MARGIN)
            slots.update(range(i0, i1 + 1))
        bursts.append((float(first), len(slots)))
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
