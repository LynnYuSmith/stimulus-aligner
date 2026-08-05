"""Decode a recording's photodiode pulse markers and align a played stimulus protocol
onto its frame-exact timeline.

The companion stimulus runner flashes a RED corner marker at each block onset, pulse-coded
by count (grey = 1, still = 2, moving = 3; a black rest screen has none). A photodiode on
the recording rig captures those pulses, and a frame-clock channel timestamps every imaging
frame. This package reads those two channels and recovers, frame-exact:

* which block each marker was (by pulse count), and
* when it started (the photodiode onset mapped through the frame clock).

Then it aligns that decoded sequence to the protocol the runner wrote, attaching each
block's label and orientation and reporting the intended-vs-true timing offset.
"""
from .decode import (
    PULSE_TO_TYPE,
    TYPE_TO_PULSE,
    decode_recording,
    detect_frame_clock,
    detect_pulse_bursts,
    time_to_frame,
)
from .align import align_to_protocol

__all__ = [
    "PULSE_TO_TYPE",
    "TYPE_TO_PULSE",
    "detect_frame_clock",
    "detect_pulse_bursts",
    "time_to_frame",
    "decode_recording",
    "align_to_protocol",
]
