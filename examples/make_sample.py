"""Synthesize a recording: a frame clock + a photodiode channel with realistic pulse markers.

Each block onset emits a marker of N pulses (grey=1, still=2, moving=3; black none). Each
pulse is a short burst of ~120 Hz flicker spanning ~0.3–1.4 V (so it crosses a mid threshold
many times — the thing a naive crossing-count gets wrong). Gratings add low-amplitude flicker
(< 0.1 V) that must be ignored. A small, random per-block acquisition lag is injected so the
decoded onsets differ from the intended ones — exactly what the aligner measures.

The `protocol` returned is the runner's played-protocol schema; the recording is built to
match it, so a correct aligner recovers it.
"""
import numpy as np

PULSES = {"grey": 1, "still": 2, "moving": 3, "black": 0}
PULSE_PERIOD_S = 6 / 51.0        # ~0.118 s, onset-to-onset (matches the runner)
PULSE_FLICKER_HZ = 120.0
PULSE_WIDTH_S = 0.030            # one pulse's flicker burst duration
FRAME_FPS = 30.0


def _protocol(blocks):
    seq, t = [], 0.0
    oris = set()
    for i, (typ, dur, ori) in enumerate(blocks):
        item = {"index": i, "type": typ,
                "label": {"grey": "Grey", "black": "Black", "moving": f"Moving {ori}°",
                          "still": f"Static {ori}°"}[typ],
                "orientation_deg": None if typ in ("grey", "black") else ori,
                "start_time_s": round(t, 3), "end_time_s": round(t + dur, 3),
                "duration_s": dur, "marker_pulses": PULSES[typ]}
        if typ in ("moving", "still"):
            oris.add(ori)
        seq.append(item)
        t += dur
    return {"protocol_name": "synthetic", "sequence": seq,
            "orientations_deg": sorted(oris), "total_duration_s": round(t, 3)}


def make_recording(blocks=None, sr=1000.0, seed=0, lag_ms=25.0):
    """Return ``(frame_chan, pulse_chan, sr, protocol, truth)``.

    ``blocks`` is a list of ``(type, duration_s, orientation)``; a default 8-orientation
    sequence with grey gaps and black pads is used if omitted. ``truth`` lists the true
    ``(type, onset_s, onset_frame)`` for every marker-bearing block.
    """
    rng = np.random.RandomState(seed)
    if blocks is None:
        blocks = [("black", 3.0, None), ("grey", 2.0, None)]
        for k, ori in enumerate([0, 45, 90, 135, 180, 225, 270, 315]):
            blocks.append(("moving", 2.0, ori))
            blocks.append(("grey", 2.0, None))
        blocks.append(("black", 3.0, None))

    proto = _protocol(blocks)
    total = proto["total_duration_s"]
    n = int(total * sr) + 1
    t = np.arange(n) / sr

    # frame clock: a short high pulse at each frame onset
    frame_chan = np.zeros(n)
    frame_onsets = np.arange(0, total, 1.0 / FRAME_FPS)
    for fo in frame_onsets:
        i = int(fo * sr)
        frame_chan[i:i + max(1, int(0.004 * sr))] = 1.0

    # photodiode: baseline + marker bursts + grating flicker + noise
    pulse_chan = np.full(n, 0.1)                       # small positive baseline
    truth = []
    for b in proto["sequence"]:
        typ, onset = b["type"], b["start_time_s"]
        npulse = PULSES[typ]
        lag = rng.uniform(0, lag_ms / 1000.0)          # per-block acquisition lag
        if npulse > 0:
            first = onset + lag
            truth.append((typ, first, int(first * FRAME_FPS)))
            for k in range(npulse):
                ps = first + k * PULSE_PERIOD_S
                seg = (t >= ps) & (t < ps + PULSE_WIDTH_S)
                flick = 0.5 * (1 + np.sign(np.sin(2 * np.pi * PULSE_FLICKER_HZ * (t - ps))))
                pulse_chan[seg] = 0.3 + 1.1 * flick[seg]   # ~0.3–1.4 V
        if typ in ("moving", "still"):                 # low-amplitude grating flicker (must be ignored)
            seg = (t >= onset) & (t < b["end_time_s"])
            pulse_chan[seg] += 0.05 * np.sin(2 * np.pi * 2.0 * (t[seg] - onset))

    pulse_chan += rng.normal(0, 0.01, n)
    frame_chan += rng.normal(0, 0.01, n)
    return frame_chan, pulse_chan, sr, proto, truth


if __name__ == "__main__":
    fc, pc, sr, proto, truth = make_recording()
    print(f"recording {len(fc)} samples @ {sr:.0f} Hz, {proto['total_duration_s']:.1f} s; "
          f"{len(truth)} marker blocks, {len(proto['sequence'])} total blocks")
