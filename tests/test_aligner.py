import numpy as np

from make_sample import make_recording
from aligner import (
    align_to_protocol,
    decode_recording,
    detect_frame_clock,
    detect_pulse_bursts,
    time_to_frame,
)


def test_frame_clock_recovers_all_frames():
    fc, _pc, sr, proto, _t = make_recording(seed=0)
    onsets = detect_frame_clock(fc, sr)
    expected = int(proto["total_duration_s"] * 30.0)   # FRAME_FPS
    # make_sample puts a frame pulse at t=0, so the count must be EXACT (no off-by-one slack):
    # the leading edge at sample 0 is a real frame and must be emitted.
    assert len(onsets) == expected
    assert onsets[0] == 0.0                              # frame 0 starts at t=0
    assert np.all(np.diff(onsets) > 0)
    assert abs(np.median(np.diff(onsets)) - 1 / 30.0) < 1e-3


def test_frame_clock_high_at_sample_zero_no_offset():
    # a recording trimmed to start ON a frame (clock high at sample 0) must not lose that edge
    sr = 1000.0
    fc = np.zeros(1000)
    for fo in np.arange(0.0, 1.0, 1 / 30.0):            # first pulse at t=0
        fc[int(fo * sr):int(fo * sr) + 4] = 1.0
    onsets = detect_frame_clock(fc, sr)
    assert onsets[0] == 0.0
    assert len(onsets) == 30


def test_pulse_counts_match_block_types():
    _fc, pc, sr, proto, truth = make_recording(seed=0)
    bursts = detect_pulse_bursts(pc, sr)
    got = [n for _o, n in bursts]
    want = [{"grey": 1, "still": 2, "moving": 3}[t[0]] for t in truth]
    assert got == want            # every marker decoded to the right pulse count


def test_grating_flicker_is_ignored():
    # a still grating (2 pulses) must not be miscounted up by its own low-amp flicker
    _fc, pc, sr, _p, _t = make_recording(blocks=[("still", 4.0, 90)], seed=1)
    bursts = detect_pulse_bursts(pc, sr)
    assert len(bursts) == 1 and bursts[0][1] == 2


def test_decode_recording_types_and_frames():
    fc, pc, sr, _proto, truth = make_recording(seed=0)
    events = decode_recording(fc, pc, sr)
    assert [e["type"] for e in events] == [t[0] for t in truth]
    # decoded onset frame EXACT (the frame clock is the ground truth; with the sample-0 edge
    # emitted there is no systematic bias — a stray <=1 slack previously hid an off-by-one)
    for e, t in zip(events, truth):
        assert e["onset_frame"] == t[2]


# ---- hard cases the adversarial review surfaced (these must not silently mis-decode) ----

def _pulse_channel(n_pulses, sr=1000.0, first=0.5, width=0.059, period=6 / 51.0,
                   amp=1.4, base=0.1, dur=3.0, bridge=False):
    """A photodiode segment with `n_pulses` marker pulses (flickering), optionally bridged
    (continuous high across pulses = monitor persistence, no interior edge)."""
    n = int(dur * sr); t = np.arange(n) / sr; pc = np.full(n, base)
    if bridge:
        pc[(t >= first) & (t < first + (n_pulses - 1) * period + width)] = amp
    else:
        for k in range(n_pulses):
            ps = first + k * period
            seg = (t >= ps) & (t < ps + width)
            fl = 0.5 * (1 + np.sign(np.sin(2 * np.pi * 120 * (t - ps))))
            pc[seg] = 0.3 + (amp - 0.3) * fl[seg]
    return pc, sr


def test_merged_pulses_still_counted():
    # two pulses with no dip between them (persistence) must NOT drop to a lower count
    pc, sr = _pulse_channel(3, bridge=True)
    b = detect_pulse_bursts(pc, sr)
    assert len(b) == 1 and b[0][1] == 3          # bridged 3-pulse moving stays "moving"


def test_pulse_counts_robust_to_lag():
    for npul in (1, 2, 3):
        pc, sr = _pulse_channel(npul, first=0.5 + 0.025)   # 25 ms acquisition lag
        assert detect_pulse_bursts(pc, sr)[0][1] == npul


def test_weak_marker_not_dropped_by_hardcoded_threshold():
    # a marker only ~0.45 V above baseline must still be found (threshold derived from signal)
    pc, sr = _pulse_channel(3, amp=0.55, base=0.1)     # peak ~0.55, base 0.1 → span ~0.45
    b = detect_pulse_bursts(pc, sr)
    assert len(b) == 1 and b[0][1] == 3


def test_bright_block_in_base_window_does_not_hide_marker():
    # a bright block occupying the first 2 s must not push the threshold above real markers
    sr = 1000.0; n = 5000; t = np.arange(n) / sr; pc = np.full(n, 0.1)
    pc[(t >= 0.1) & (t < 1.7)] = 1.5                    # bright block inside the old base window
    for k in range(3):                                  # a real moving marker at 2.5 s
        ps = 2.5 + k * (6 / 51.0)
        pc[(t >= ps) & (t < ps + 0.059)] = 1.4
    b = detect_pulse_bursts(pc, sr)
    assert any(cnt == 3 for _o, cnt in b)              # the marker survives


def test_nan_in_frame_channel_no_phantom_edge():
    sr = 1000.0
    fc = np.zeros(2000)
    for fo in np.arange(0.0, 2.0, 1 / 30.0):
        fc[int(fo * sr):int(fo * sr) + 8] = 1.0
    clean = len(detect_frame_clock(fc, sr))
    fc[int(0.5 * sr) + 3] = np.nan                      # dropped sample mid-pulse
    assert len(detect_frame_clock(fc, sr)) == clean     # no phantom split edge


def test_ok_gate_rejects_large_offset_by_default():
    # default tolerance must NOT be disabled: a marker far from its intended time fails ok
    proto = {"sequence": [{"type": "moving", "label": "Moving 0°",
                           "orientation_deg": 0, "start_time_s": 0.0}]}
    ev = [{"type": "moving", "n_pulses": 3, "onset_s": 999.0, "onset_frame": 30000}]
    assert align_to_protocol(ev, proto)["ok"] is False           # default tol catches it
    assert align_to_protocol(ev, proto, tolerance_s=None, spread_s=None)["ok"] is True


def test_ok_gate_rejects_shifted_all_moving_alignment():
    # an all-moving protocol with a dropped + spurious marker keeps equal count and all types,
    # but the per-block offsets are non-constant → the spread check must reject it
    seq, t = [], 0.0
    for i, o in enumerate([0, 45, 90, 135]):
        seq.append({"type": "moving", "label": f"Moving {o}°", "orientation_deg": o,
                    "start_time_s": t}); t += 4.0
    proto = {"sequence": seq}
    # events shifted by one block (each decoded onset is really the NEXT block's time)
    ev = [{"type": "moving", "n_pulses": 3, "onset_s": b["start_time_s"] + 4.0,
           "onset_frame": 0} for b in seq]
    r = align_to_protocol(ev, proto)
    assert r["type_agreement"] == 1.0 and r["ok"] is False        # types agree but ok rejects


def test_time_to_frame_edges():
    onsets = np.array([0.0, 0.1, 0.2, 0.3])
    assert time_to_frame(-1.0, onsets) == 0
    assert time_to_frame(0.15, onsets) == 1
    assert time_to_frame(9.0, onsets) == 3
    assert time_to_frame(0.0, np.array([])) == -1


def test_align_recovers_protocol():
    fc, pc, sr, proto, _t = make_recording(seed=0, lag_ms=25.0)
    events = decode_recording(fc, pc, sr)
    res = align_to_protocol(events, proto, tolerance_s=0.05)
    assert res["ok"]
    assert res["n_matched"] == res["n_blocks"]      # every marker block matched
    assert res["type_agreement"] == 1.0
    # the injected lag is positive and small → decoded onsets trail the intended ones
    assert 0.0 <= res["median_offset_s"] <= 0.05
    # labels + orientations carried through
    movings = [a for a in res["aligned"] if a["type"] == "moving"]
    assert [a["orientation_deg"] for a in movings] == [0, 45, 90, 135, 180, 225, 270, 315]
    assert movings[1]["label"] == "Moving 45°"


def test_unknown_pulse_count_is_visible_not_dropped():
    # four pulses (a mis-generated marker) should surface as "?4", not vanish
    from make_sample import PULSE_PERIOD_S
    sr = 1000.0
    n = int(3.0 * sr)
    pc = np.full(n, 0.1)
    for k in range(4):
        s = int((0.5 + k * PULSE_PERIOD_S) * sr)
        pc[s:s + int(0.03 * sr)] = 1.3
    fc = np.zeros(n)
    for fo in np.arange(0, 3.0, 1 / 30.0):
        fc[int(fo * sr):int(fo * sr) + 4] = 1.0
    events = decode_recording(fc, pc, sr)
    assert len(events) == 1 and events[0]["type"] == "?4"
