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
    assert abs(len(onsets) - expected) <= 1
    # onsets are monotonic and ~1/30 s apart
    assert np.all(np.diff(onsets) > 0)
    assert abs(np.median(np.diff(onsets)) - 1 / 30.0) < 1e-3


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
    # decoded onset frame within 1 frame of truth
    for e, t in zip(events, truth):
        assert abs(e["onset_frame"] - t[2]) <= 1


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
