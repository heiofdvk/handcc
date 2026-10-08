# HandCC

Webcam hand tracking to MIDI CC on macOS: hand gestures become control changes on a virtual MIDI port that Ableton Live (or any MIDI host) can map.

## What it does and why

Expressive controls such as the mod wheel, expression, breath or a filter cutoff usually need hardware: a wheel, a pedal, a breath controller. HandCC uses the laptop webcam instead. It tracks both hands with Google's MediaPipe hand landmarker, measures things like how high a hand is, how far the thumb and index finger are apart or how open the hand is, and sends each one as a MIDI CC on a virtual port called **HandCC**.

Each control is one entry in `mappings.toml`, so a setup such as "right-hand height to CC11 (expression), left-hand pinch to CC1 (mod wheel)" is a few lines of text. A preview window shows the tracked hands and the current value of every mapping.

## How it works

1. **Capture.** OpenCV reads frames from the camera (AVFoundation backend, 1280x720 by default). With `mirror = true` the frame is flipped horizontally, so the preview behaves like a mirror.
2. **Landmarks.** MediaPipe's hand landmarker (running on the CPU) returns, for up to two hands, 21 landmarks in image coordinates, the same 21 points as 3D "world" coordinates in metres centred on the hand, and a Left/Right label.
3. **Features.** Each hand is reduced to eight numbers:

   | feature  | what it measures | default raw range |
   |----------|------------------|-------------------|
   | `x`      | palm centre, left to right of the frame | 0.15 to 0.85 |
   | `y`      | palm centre, bottom to top of the frame | 0.15 to 0.85 |
   | `depth`  | wrist to middle-knuckle length on screen (larger when the hand is closer) | 0.12 to 0.35 |
   | `tilt`   | hand turned like a dial, in degrees (0 = fingers pointing up) | -60 to 60 |
   | `pinch`  | thumb tip to index tip distance | 0.15 to 1.0 |
   | `pinch2` | thumb tip to middle tip distance | 0.2 to 1.2 |
   | `spread` | index tip to pinky tip distance (fingers together to splayed) | 0.5 to 1.3 |
   | `open`   | mean wrist to fingertip distance (fist to open hand) | 0.9 to 1.9 |

   `x`, `y`, `depth` and `tilt` come from the image landmarks. The shape features (`pinch`, `pinch2`, `spread`, `open`) use the 3D world landmarks divided by the palm length (wrist to middle knuckle), so they change little when the hand turns or moves closer or further away.
4. **Scaling.** Each mapping maps its feature's raw range (the default above, a `range` from `mappings.toml`, or a calibrated range) to 0..1, clamps it, optionally inverts it and applies a `curve` exponent.
5. **Smoothing.** A [One Euro filter](https://gery.casiez.net/1euro/) (Casiez, Roussel and Vogel, CHI 2012) smooths each mapping. It filters heavily when the hand is still and lightly when it moves fast, so a resting hand does not jitter and quick gestures are not smeared. The result is scaled to 0..127.
6. **Hysteresis.** A change of exactly one step that reverses the previous direction is dropped (except at 0 and 127), so a still hand does not flicker between two adjacent values. Repeated values are never sent.
7. **Output.** Values go out as MIDI control change messages through [mido](https://github.com/mido/mido) and [python-rtmidi](https://github.com/SpotlightKid/python-rtmidi), on a virtual CoreMIDI port that the script creates. If an existing MIDI output contains the configured `port` name (for example `IAC`), that output is used instead.

**Hand loss.** Tracking often drops single frames, so a hand counts as gone only after 0.25 s. Then the mapping holds its last value, or sends `on_lost` if the mapping defines it.

**Handedness and mirroring.** Mappings refer to your real left and right hands. With MediaPipe 0.10.21, the Left/Right labels were correct on the raw, unflipped camera frame. Flipping the frame for the mirror view also flips the labels, so when `mirror = true` the script swaps them back. If the preview still labels your hands backwards, set `swap_hands = true`. With the mirror view, moving a hand to your right raises its `x`.

**Calibration.** The default ranges suit an average hand at a typical laptop distance. Press `c` in the preview window, move each control through its full range, and press `c` again. For every mapping with enough samples, the 3rd and 97th percentiles of what it saw become its new range (percentiles, so one bad frame cannot stretch it). A mapping that barely moved keeps its old range, because a very narrow range would make the CC jump from 0 to 127 on a twitch. Ranges are saved to `calibration.json` by mapping name and take precedence over `range` in `mappings.toml`. Solo a mapping first to calibrate only that one. `x` deletes the calibration.

## Requirements

- **macOS.** The camera code uses AVFoundation (through OpenCV and PyObjC), and the virtual MIDI port uses CoreMIDI. Developed and tested on an Apple Silicon Mac. MediaPipe 0.10.21 also publishes Intel Mac wheels, but Intel Macs have not been tested.
- **Python 3.11 or 3.12.** The script needs `tomllib` (3.11+), and MediaPipe 0.10.21 has wheels up to 3.12. Developed with 3.12.
- **mediapipe 0.10.21 (pinned).** Version 1.0.1 aborted on the development Mac inside MediaPipe's Metal helper (`DrishtiMetalHelper`), even with the CPU delegate selected. 0.10.21 works with the CPU delegate. It requires `numpy<2`, hence the NumPy 1.26 pin.
- **mido + python-rtmidi** for the virtual MIDI port, **OpenCV** for the camera and preview window, **PyObjC (AVFoundation)** to request camera permission. Exact versions are in `requirements.txt`.
- A webcam, and a MIDI host such as Ableton Live.

## Install

```sh
git clone <this repository> handcc
cd handcc
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The environment has to be `.venv` inside the project folder, because `run.sh` and the app wrapper start `.venv/bin/python`. With [uv](https://docs.astral.sh/uv/): `uv venv --python 3.12 && uv pip install -r requirements.txt`.

**Model.** On the first run, HandCC downloads MediaPipe's hand landmarker model (`hand_landmarker.task`, 7.8 MB, Apache-2.0) from Google's model storage into the project folder. To fetch it yourself instead:

```sh
curl -L -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
shasum -a 256 hand_landmarker.task
# fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1
```

## Usage

### Camera permission and the app wrapper

macOS only lets an app use the camera after the user allows it, and it asks on behalf of the app that started the process. On the development Mac, Terminal.app never showed that prompt: the request was denied silently. HandCC.app is a small wrapper that works around this. Its `Info.plist` contains a camera usage description, so macOS shows the prompt for HandCC, and its launcher runs `handcc.py` with the project's `.venv`.

Build it once (the files it is made from are in `app/`):

```sh
./make_app.sh       # creates HandCC.app in the project folder and signs it ad hoc
open HandCC.app     # or double-click it in Finder
```

- Allow camera access when asked. To reset the decision: `tccutil reset Camera local.handcc`.
- Output (status lines, solo changes, calibration results, errors) goes to `handcc.log` in the project folder, overwritten on each launch.
- Keep HandCC.app in the project folder: the launcher finds `.venv` and `handcc.py` relative to the app's own location.
- Rebuilding the app changes its ad hoc signature, so macOS may ask for camera access again.

### From a terminal

A terminal that already has camera permission (for example the integrated terminal of VS Code) can run the script directly:

```sh
./run.sh                    # preview window + MIDI out
./run.sh --list-cameras     # list camera indices (camera 0 may be an iPhone via Continuity Camera)
./run.sh --camera 1         # use another camera
./run.sh --list-ports       # list existing MIDI outputs
./run.sh --config my.toml   # use another mappings file
./run.sh --image hands.jpg  # no camera: print the features for a still image, save hands_annotated.jpg
```

`run.sh` hides MediaPipe's start-up log lines. `--image` is useful for checking the install and the feature values without a camera.

### Keys (preview window focused)

| key     | action |
|---------|--------|
| `1`-`9` | solo that mapping: only it sends. Use this while MIDI-mapping in the host |
| `0`     | all mappings send again |
| `space` | freeze / unfreeze all output |
| `c`     | start calibrating; press again to save (see Calibration above) |
| `x`     | clear the calibration |
| `a`     | re-send every current value |
| `q`, `Esc` | quit |

### Ableton Live

1. Start HandCC **before** Ableton, or reopen Settings > Link, Tempo & MIDI after starting it, so the HandCC port is listed.
2. Under Input Ports, turn on **Remote** for HandCC to MIDI-map controls. Turn on **Track** as well to send the CCs into an instrument on an armed or monitored track (for example CC1 mod wheel, CC11 expression, CC74 brightness).
3. Press Cmd+M, click a control, press the mapping's number key in the HandCC window (solo) and move that hand. Press `0` when done.

The port is an ordinary CoreMIDI source, so other MIDI hosts can use it the same way.

## mappings.toml

Global settings come first, then one `[[map]]` table per control:

```toml
port = "HandCC"      # virtual port to create, or part of the name of an existing output (e.g. "IAC")
channel = 1          # default MIDI channel, 1-16
camera = 0           # camera index (see --list-cameras)
mirror = true        # mirror view: move right, the hand moves right on screen
swap_hands = false   # true if the preview labels your hands backwards

[smoothing]          # One Euro filter defaults for all mappings
min_cutoff = 1.0     # lower = steadier when still, but more lag
beta = 2.0           # higher = follows fast moves more closely

[[map]]
name = "R height"    # label in the window; must be unique (calibration is stored by name)
hand = "Right"       # "Left", "Right" or "Any" (your real hands)
feature = "y"
cc = 11              # expression

[[map]]
name = "L pinch"
hand = "Left"
feature = "pinch"
cc = 1               # mod wheel
invert = true        # pinched shut = 127
curve = 1.5          # more resolution near 0
range = [0.2, 0.9]   # raw feature range mapped to 0..127
on_lost = 0          # send 0 when the left hand leaves the frame
```

Per-mapping keys:

| key | meaning | default |
|-----|---------|---------|
| `feature` | one of `x`, `y`, `depth`, `tilt`, `pinch`, `pinch2`, `spread`, `open` | required |
| `cc` | controller number, 0-127 | required |
| `hand` | `Left`, `Right` or `Any` (`Any` uses the first hand found) | `Any` |
| `name` | label in the window and key for calibration | `<hand> <feature>` |
| `channel` | MIDI channel, 1-16 | global `channel` |
| `range` | `[lo, hi]` raw feature range mapped to 0..127 | per feature, see the table above |
| `invert` | flip the direction | `false` |
| `curve` | exponent: 1 = linear, above 1 = more resolution at the low end, below 1 = at the top | `1.0` |
| `on_lost` | CC value to send when the hand leaves the frame | hold the last value |
| `min_cutoff`, `beta` | per-mapping smoothing | `[smoothing]` values |

Optional global keys that are not in the shipped file: `width` and `height` (capture size, default 1280x720), `num_hands` (default 2; two hands are always tracked when mappings use both), and `detection_confidence`, `presence_confidence`, `tracking_confidence` (MediaPipe thresholds, defaults 0.6, 0.5, 0.5).

The shipped `mappings.toml` sends right-hand pinch, height and tilt on CC2, CC11 and CC74, and left-hand height, openness and finger spread on CC1, CC21 and CC22.

## Limitations

- macOS only as written (AVFoundation capture, PyObjC permission check, CoreMIDI virtual port). Other platforms would need changes to those parts.
- Tracking depends on light, background and camera. Hands that overlap, cross or leave the frame lose tracking. If both hands get the same label in a frame, the second is ignored.
- Latency is the sum of the camera frame time, CPU inference and the smoothing filter. The GPU delegate is not used because it crashed on macOS.
- CC only, 7-bit: no notes, no pitch bend, no 14-bit CC. The hysteresis drops single-step reversals, which costs a little resolution on very slow movements.
- `depth` is the on-screen size of the palm, so it also changes when the hand tilts toward or away from the camera.
- Only the first nine mappings can be soloed from the keyboard.
- Calibration is stored by mapping name, so renaming a mapping discards its calibration.
- HandCC.app is ad hoc signed, meant to be built locally, and must stay in the project folder.
- The handedness swap relies on the labelling behaviour observed with MediaPipe 0.10.21. `swap_hands` is there in case another version or camera behaves differently.

## Credits and licences

Written by Joaquín Jacubowicz, a musician and producer. Built with [Claude Code](https://claude.com/claude-code): the author designed the tool, directed the agent and tested the results.

- [MediaPipe](https://github.com/google-ai-edge/mediapipe) (Google): Apache-2.0. The hand landmarker model (`hand_landmarker.task`, see the [Hand Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)) is also Apache-2.0. It is downloaded at first run and is not part of this repository.
- One Euro filter: G. Casiez, N. Roussel, D. Vogel, "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems", CHI 2012. Implemented in `handcc.py` (class `OneEuro`).
- OpenCV (`opencv-contrib-python`): Apache-2.0. The prebuilt wheels bundle third-party libraries under their own licences.
- mido: MIT. python-rtmidi: MIT (includes RtMidi, MIT-style licence).
- NumPy: BSD-3-Clause. PyObjC: MIT.
