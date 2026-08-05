"""Align decoded marker events to a played protocol from the runner."""
from __future__ import annotations

import numpy as np


def _marker_blocks(protocol):
    """The protocol's marker-bearing blocks (grey/still/moving), in order — black has none."""
    seq = protocol.get("sequence", protocol) if isinstance(protocol, dict) else protocol
    return [b for b in seq if b.get("type") in ("grey", "still", "moving")]


def align_to_protocol(events, protocol, *, tolerance_s=0.5, spread_s=0.25):
    """Match decoded events to the played protocol, in order, and attach its labels.

    ``events`` is the output of :func:`decode_recording`; ``protocol`` is the runner's
    ``protocol_played.json`` (or its ``sequence`` list). Matching is positional over the
    marker-bearing blocks (grey/still/moving), which is unambiguous because both sequences
    are ordered and the black rest screens carry no marker on either side.

    ``tolerance_s`` bounds the acceptable |decoded − intended| onset offset; the default is a
    real bound (not ``None``), so a positional mis-pairing that happens to keep types (an
    all-moving protocol with a dropped + spurious marker) is caught by its large offset rather
    than passing silently. ``spread_s`` bounds how much the per-block offsets may vary: a
    correct alignment has a roughly CONSTANT offset (shared acquisition lag), so an offset
    *spread* wider than ``spread_s`` means the events are paired to the wrong blocks even if
    every offset is individually small. Set either to ``None`` to disable that check.

    Returns a dict:
      * ``aligned`` — one row per matched pair with ``type``, decoded ``onset_s`` /
        ``onset_frame``, the protocol ``label`` / ``orientation_deg`` / intended
        ``start_time_s``, and ``offset_s`` (decoded − intended).
      * ``n_events`` / ``n_blocks`` / ``n_matched``
      * ``type_agreement`` — fraction of matched pairs whose decoded type equals the protocol's
      * ``median_offset_s`` / ``max_abs_offset_s`` / ``offset_spread_s`` over matched pairs
      * ``ok`` — True iff every block matched, every type agreed, |offset| ≤ ``tolerance_s``,
        and the offset spread ≤ ``spread_s``.
    """
    blocks = _marker_blocks(protocol)
    n = min(len(events), len(blocks))
    aligned, offsets, type_ok = [], [], 0
    for i in range(n):
        ev, bl = events[i], blocks[i]
        intended = bl.get("start_time_s")
        offset = None if intended is None else ev["onset_s"] - float(intended)
        agree = ev["type"] == bl.get("type")
        type_ok += int(agree)
        if offset is not None:
            offsets.append(offset)
        aligned.append({
            "type": ev["type"],
            "type_matches": agree,
            "onset_s": ev["onset_s"],
            "onset_frame": ev["onset_frame"],
            "label": bl.get("label"),
            "orientation_deg": bl.get("orientation_deg"),
            "intended_start_s": None if intended is None else float(intended),
            "offset_s": offset,
        })
    offs = np.asarray(offsets, dtype=float) if offsets else np.array([])
    median_off = float(np.median(offs)) if offs.size else None
    max_abs = float(np.max(np.abs(offs))) if offs.size else None
    spread = float(np.max(offs) - np.min(offs)) if offs.size else None
    all_matched = len(events) == len(blocks)
    all_types = type_ok == n and n > 0
    # a real alignment has a small, roughly-constant offset; check both magnitude and spread
    within_tol = tolerance_s is None or (max_abs is not None and max_abs <= tolerance_s)
    within_spread = spread_s is None or (spread is not None and spread <= spread_s)
    return {
        "aligned": aligned,
        "n_events": len(events),
        "n_blocks": len(blocks),
        "n_matched": n,
        "type_agreement": (type_ok / n) if n else 0.0,
        "median_offset_s": median_off,
        "max_abs_offset_s": max_abs,
        "offset_spread_s": spread,
        "ok": bool(all_matched and all_types and within_tol and within_spread),
    }
