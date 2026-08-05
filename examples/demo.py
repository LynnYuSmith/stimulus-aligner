"""Show the point: the runner's INTENDED (wall-clock) onsets carry acquisition lag; decoding
the recording's photodiode markers recovers the TRUE, frame-exact onsets — and aligns the
played protocol onto them (right type, right orientation, in order)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_sample import make_recording
from aligner import align_to_protocol, decode_recording, detect_frame_clock

CTrue, CInt = "#6b7f9e", "#9e7b8a"   # muted slate (decoded) / mauve (intended)


def main() -> None:
    fc, pc, sr, proto, _truth = make_recording(seed=0, lag_ms=25.0)
    events = decode_recording(fc, pc, sr)
    res = align_to_protocol(events, proto, tolerance_s=0.05)
    t = np.arange(len(pc)) / sr

    fig, ax = plt.subplots(2, 1, figsize=(9.2, 4.6),
                           gridspec_kw={"height_ratios": [2, 1]})

    # top: the photodiode trace with intended vs decoded onsets
    a0 = ax[0]
    a0.plot(t, pc, color="#bdbdbd", lw=0.6)
    for a in res["aligned"]:
        a0.axvline(a["intended_start_s"], color=CInt, lw=1.0, ls="--", alpha=0.8)
        a0.axvline(a["onset_s"], color=CTrue, lw=1.0, alpha=0.9)
    a0.plot([], [], color=CInt, ls="--", label="intended onset (runner log)")
    a0.plot([], [], color=CTrue, label="decoded onset (photodiode, frame-exact)")
    a0.set_ylabel("photodiode, V")
    a0.set_title("stimulus aligner: decode the markers, recover the true onsets, "
                 "align the played protocol", fontsize=10)
    a0.legend(fontsize=7, loc="upper right", framealpha=0.9)
    a0.set_xlim(0, proto["total_duration_s"])

    # bottom: per-block timing offset (decoded − intended)
    a1 = ax[1]
    idx = np.arange(len(res["aligned"]))
    offs = [a["offset_s"] * 1000 for a in res["aligned"]]
    labels = [a["label"] for a in res["aligned"]]
    a1.bar(idx, offs, color=CTrue, width=0.7)
    a1.axhline(0, color="#9a9a9a", lw=0.8)
    a1.set_xticks(idx)
    a1.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    a1.set_ylabel("offset, ms")
    a1.set_xlabel("block (in order)")
    fig.tight_layout()

    out = Path(__file__).resolve().parent.parent / "figures" / "before_after.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130)

    print(f"blocks {res['n_blocks']}, matched {res['n_matched']}, "
          f"type agreement {res['type_agreement']*100:.0f}%, "
          f"median offset {res['median_offset_s']*1000:.1f} ms, "
          f"max |offset| {res['max_abs_offset_s']*1000:.1f} ms, ok={res['ok']}")


if __name__ == "__main__":
    main()
