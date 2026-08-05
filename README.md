# stimulus-aligner

Decode a recording's photodiode **pulse markers** and align a played stimulus protocol onto
its **frame-exact** timeline.

The companion [stimulus-runner](https://github.com/LynnYuSmith/stimulus-runner) flashes a RED
corner marker at each block onset, pulse-coded by count (grey = 1, still = 2, moving = 3; a
black rest screen has none). On the rig a photodiode captures those pulses and a frame-clock
channel timestamps every imaging frame. This tool reads those two channels and recovers, for
each marker, **which block it was** (by pulse count) and **when it truly started** (the
photodiode onset mapped through the frame clock) — then aligns that onto the protocol the
runner wrote, attaching each block's label and orientation and reporting the intended-vs-true
timing offset.

![decode the markers, recover the true onsets, align the played protocol](figures/before_after.png)

## The idea

The runner's log records the *intended* (wall-clock) onset of each block, but there is
acquisition lag between "the browser drew the frame" and "the photodiode saw it." The
recording itself is the ground truth: the photodiode marker is when the stimulus *actually*
appeared, and the frame clock says which imaging frame that was.

Recovering it has one real subtlety. A photodiode marker is not a clean step — each logical
pulse is a short burst of ~120 Hz monitor flicker spanning ~0.3–1.4 V, so one pulse crosses
any fixed threshold many times. Counting raw crossings over-counts (a 3-pulse "moving" marker
would read as many more). The decoder instead groups the supra-threshold run onsets into
*pulses* by their onset-to-onset spacing, keyed on the stable inter-pulse period rather than
the width-dependent end-to-start gap — so the count stays right even when a block is
acquisition-lagged. A grating adds only low-amplitude flicker (< ~0.1 V) that never reaches
the marker threshold, so it is ignored.

## Use

```python
from aligner import decode_recording, align_to_protocol
import json

# frame_chan, pulse_chan: 1-D arrays from the recording; sr: sample rate (Hz)
events = decode_recording(frame_chan, pulse_chan, sr)
#   -> [{type, n_pulses, onset_s, onset_frame}, ...]  (grey/still/moving, in time order)

protocol = json.load(open("protocol_played.json"))     # written by the runner
result = align_to_protocol(events, protocol, tolerance_s=0.05)
result["ok"]                 # every block matched, every type agreed, within tolerance
result["median_offset_s"]    # intended-vs-true timing offset
result["aligned"]            # per block: decoded onset_frame + label + orientation_deg
```

Reading a LabChart-style `.mat` (channels + sample rate) needs `scipy`; the core decode/align
works on plain NumPy arrays, so nothing but NumPy is required for the logic and tests.

## Run the example

```bash
python -m venv .venv && source .venv/bin/activate    # Python 3.10+
pip install numpy && pip install pytest matplotlib   # matplotlib only for the figure
python examples/demo.py        # writes figures/before_after.png
pytest
```

## What's here

- `aligner/decode.py` — frame-clock detection, pulse-burst counting (the flicker-grouping),
  and `decode_recording` → frame-exact events.
- `aligner/align.py` — `align_to_protocol`: positional match to the runner's played protocol,
  with labels, orientations, and the timing offset.
- `examples/make_sample.py` — a synthetic recording (frame clock + photodiode with realistic
  flicker markers, grating flicker to ignore, and a per-block acquisition lag). Ground truth
  known.
- `examples/demo.py` — decode + align the synthetic recording → `figures/before_after.png`.
- `tests/test_aligner.py` — `pytest` (7 checks): pulse counts, ignored grating flicker,
  frame mapping, full alignment, and that an unknown pulse count surfaces rather than vanishes.

## Verification

Fully tested headless on synthetic recordings (`pytest`) — including that a mis-generated
4-pulse marker surfaces as `?4` instead of being silently dropped. On real recordings the
photodiode is the timing ground truth by design.

## License

MIT — see [LICENSE](LICENSE).
