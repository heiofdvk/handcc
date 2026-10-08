#!/bin/zsh
# Start handcc: webcam hand tracking → MIDI CC on the "HandCC" virtual port.
cd "${0:A:h}" && exec .venv/bin/python handcc.py "$@" 2> >(grep -v -E '^(I0000|W0000|INFO|WARNING: All log)' >&2)
