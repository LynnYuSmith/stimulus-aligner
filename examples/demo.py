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

    fig = plt.figure(figsize=(10, 5.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.55, wspace=0.35)
    a0 = fig.add_subplot(gs[0, :])
    a1 = fig.add_subplot(gs[1, :2])
    a2 = fig.add_subplot(gs[1, 2])

    # A: the whole recording: photodiode, block shading, decoded onsets and their labels
    a0.plot(t, pc, color="#9a9a9a", lw=0.5)
    ymax = float(np.nanmax(pc))
    for b in proto["sequence"]:
        if b["type"] in ("moving", "still"):
            a0.axvspan(b["start_time_s"], b["end_time_s"], color=CTrue, alpha=0.08, lw=0)
    for a in res["aligned"]:
        a0.axvline(a["onset_s"], color=CTrue, lw=0.8, alpha=0.9)
        if a["type"] != "grey":
            a0.text(a["onset_s"] + 1.0, ymax * 1.08, f'{a["orientation_deg"]}°',
                    fontsize=7, ha="center", va="bottom", color=CTrue)
    a0.set_ylim(-0.05, ymax * 1.25)
    a0.set_xlim(0, proto["total_duration_s"])
    a0.set_ylabel("photodiode, V")
    a0.set_xlabel("time, s")
    a0.set_title("A  every marker decoded: type by pulse count, orientation from the played protocol",
                 fontsize=9, loc="left")

    # B: one block up close: the intended onset vs the true one, on the frame clock
    k = next(i for i, a in enumerate(res["aligned"]) if a["type"] == "moving")
    a = res["aligned"][k]
    t_int, t_on = a["intended_start_s"], a["onset_s"]
    lo, hi = t_int - 0.06, t_int + 0.42
    m = (t >= lo) & (t <= hi)
    a1.plot(t[m], pc[m], color="#9a9a9a", lw=0.7)
    clock = detect_frame_clock(fc, sr)
    ft = np.asarray(clock, dtype=float)
    ft = ft[(ft >= lo) & (ft <= hi)]
    a1.vlines(ft, -0.12, -0.04, color="#555555", lw=0.8)
    a1.text(lo + 0.004, -0.02, "frame clock", fontsize=7, ha="left", va="bottom", color="#555555")
    a1.axvline(t_int, color=CInt, lw=1.4, ls="--")
    a1.axvline(t_on, color=CTrue, lw=1.4)
    a1.annotate("", xy=(t_on, ymax * 1.05), xytext=(t_int, ymax * 1.05),
                arrowprops=dict(arrowstyle="->", color="#333333", lw=0.9))
    a1.text((t_int + t_on) / 2, ymax * 1.1, f"{(t_on - t_int) * 1000:.0f} ms",
            fontsize=8, ha="center", va="bottom")
    a1.text(t_int, ymax * 0.55, "intended\n(runner log) ", fontsize=7, ha="right", color=CInt)
    a1.text(t_on, ymax * 0.55, f" true onset\n frame {a['onset_frame']}", fontsize=7,
            ha="left", color=CTrue)
    a1.set_xlim(lo, hi)
    a1.set_ylim(-0.15, ymax * 1.3)
    a1.set_xlabel("time, s")
    a1.set_ylabel("photodiode, V")
    a1.set_title(f'B  one block up close ({a["label"] if "label" in a else a["orientation_deg"]}): '
                 f'3 pulses = moving', fontsize=9, loc="left")

    # C: the offset per block: what the log would have got wrong
    off = np.array([(x["onset_s"] - x["intended_start_s"]) * 1000 for x in res["aligned"]])
    a2.plot(np.arange(len(off)), off, "o", ms=3.5, color=CTrue)
    a2.axhline(res["median_offset_s"] * 1000, color=CInt, lw=1, ls="--")
    a2.set_ylim(0, max(off.max() * 1.4, 1))
    a2.set_xlabel("block")
    a2.set_ylabel("true − intended, ms")
    a2.set_title(f'C  offset per block (median {res["median_offset_s"]*1000:.0f} ms)',
                 fontsize=9, loc="left")
    for ax in (a0, a1, a2):
        ax.spines[["top", "right"]].set_visible(False)

    out = Path(__file__).resolve().parent.parent / "figures" / "before_after.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.15)

    print(f"blocks {res['n_blocks']}, matched {res['n_matched']}, "
          f"type agreement {res['type_agreement']*100:.0f}%, "
          f"median offset {res['median_offset_s']*1000:.1f} ms, "
          f"max |offset| {res['max_abs_offset_s']*1000:.1f} ms, ok={res['ok']}")


if __name__ == "__main__":
    main()
