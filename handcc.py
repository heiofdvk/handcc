#!/usr/bin/env python3
"""handcc — webcam hand tracking → MIDI CC.

MediaPipe finds 21 landmarks per hand. Each [[map]] in mappings.toml turns one
hand feature (pinch, height, tilt, ...) into a CC, smoothed with a One Euro
filter so it sits still when your hand does and still follows fast moves.

Keys (preview window focused):
  1-9    solo that mapping (only it sends — use this while MIDI-mapping in Ableton)
  0      un-solo, all mappings send
  space  freeze / unfreeze output
  c      start calibrating; move through the full range; c again to save
  x      clear calibration (back to the ranges in mappings.toml)
  a      re-send every current value
  q/esc  quit
"""
import argparse
import json
import math
import shutil
import sys
import time
import tomllib
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import mido
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

HERE = Path(__file__).resolve().parent
MODEL = HERE / "hand_landmarker.task"
# Google's MediaPipe hand landmarker model (Apache-2.0); fetched on first run.
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
CALIB = HERE / "calibration.json"

WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP = 0, 4, 5, 8
MIDDLE_MCP, MIDDLE_TIP, RING_TIP, PINKY_MCP, PINKY_TIP = 9, 12, 16, 17, 20
FINGERTIPS = (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]

# Raw range each feature typically covers; mapped to 0..127. Override per
# mapping with `range = [lo, hi]`, or press c to calibrate to your own hand.
DEFAULT_RANGE = {
    "x": (0.15, 0.85),      # palm centre, left → right of the frame
    "y": (0.15, 0.85),      # palm centre, bottom → top
    "depth": (0.12, 0.35),  # palm size on screen: far → close to the camera
    "tilt": (-60.0, 60.0),  # degrees, hand turned like a dial (0 = fingers up)
    "pinch": (0.15, 1.0),   # thumb–index gap, relative to palm length
    "pinch2": (0.2, 1.2),   # thumb–middle gap
    "spread": (0.5, 1.3),   # index–pinky tip gap: fingers together → splayed
    "open": (0.9, 1.9),     # fist → open hand
}
LOST_GRACE_S = 0.25  # tracking drops single frames; don't treat that as "hand gone"


class OneEuro:
    """One Euro filter (Casiez et al. 2012): low jitter at rest, low lag in motion."""

    def __init__(self, min_cutoff, beta, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.t = self.x = None
        self.dx = 0.0

    @staticmethod
    def _alpha(dt, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.t is None:
            self.t, self.x = t, x
            return x
        dt = max(t - self.t, 1e-4)
        a_d = self._alpha(dt, self.d_cutoff)
        self.dx = a_d * (x - self.x) / dt + (1 - a_d) * self.dx
        a = self._alpha(dt, self.min_cutoff + self.beta * abs(self.dx))
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x


def hand_features(img_lm, world_lm, aspect):
    """Features in raw units (see DEFAULT_RANGE). Shape features use MediaPipe's
    3D world landmarks so they don't change as the hand turns or moves away."""
    P = np.array([(p.x * aspect, p.y) for p in img_lm])  # image space, frame height = 1
    W = np.array([(p.x, p.y, p.z) for p in world_lm])    # metres, hand-centred

    def dist(A, i, j):
        return float(np.linalg.norm(A[i] - A[j]))

    palm = dist(W, WRIST, MIDDLE_MCP) or 1e-6
    cx, cy = np.mean([(img_lm[i].x, img_lm[i].y) for i in (WRIST, INDEX_MCP, MIDDLE_MCP, 13, PINKY_MCP)], axis=0)
    vx, vy = P[MIDDLE_MCP] - P[WRIST]
    return {
        "x": float(cx),
        "y": float(1 - cy),
        "depth": dist(P, WRIST, MIDDLE_MCP),
        "tilt": math.degrees(math.atan2(vx, -vy)),
        "pinch": dist(W, THUMB_TIP, INDEX_TIP) / palm,
        "pinch2": dist(W, THUMB_TIP, MIDDLE_TIP) / palm,
        "spread": dist(W, INDEX_TIP, PINKY_TIP) / palm,
        "open": float(np.mean([dist(W, WRIST, t) for t in FINGERTIPS])) / palm,
    }


class Mapping:
    def __init__(self, spec, cfg, calibration):
        self.feature = spec["feature"]
        if self.feature not in DEFAULT_RANGE:
            sys.exit(f"Unknown feature {self.feature!r}. Choose from: {', '.join(DEFAULT_RANGE)}")
        self.hand = spec.get("hand", "any").capitalize()
        if self.hand not in ("Left", "Right", "Any"):
            sys.exit(f"hand must be Left, Right or Any (got {spec['hand']!r})")
        self.cc = int(spec["cc"])
        self.name = spec.get("name") or f"{self.hand} {self.feature}"
        self.channel = int(spec.get("channel", cfg.get("channel", 1))) - 1
        self.config_range = tuple(spec.get("range", DEFAULT_RANGE[self.feature]))
        self.range = tuple(calibration.get(self.name, self.config_range))
        self.invert = bool(spec.get("invert", False))
        self.curve = float(spec.get("curve", 1.0))
        self.on_lost = spec.get("on_lost")  # None = hold last value
        sm = cfg.get("smoothing", {})
        self.filter = OneEuro(float(spec.get("min_cutoff", sm.get("min_cutoff", 1.0))),
                              float(spec.get("beta", sm.get("beta", 2.0))))
        self.raw = None       # latest raw feature value
        self.value = None     # last CC value sent
        self.direction = 0    # sign of the last change sent (for hysteresis)
        self.display = None   # current CC value, sent or not (for the HUD)
        self.last_seen = 0.0
        self.samples = []     # raw values collected while calibrating

    def normalise(self, raw):
        lo, hi = self.range
        v = (raw - lo) / (hi - lo) if hi != lo else 0.0
        v = min(max(v, 0.0), 1.0)
        if self.invert:
            v = 1.0 - v
        return v ** self.curve

    def update(self, raw, t):
        """Return the current CC value (send() skips repeats)."""
        self.raw, self.last_seen = raw, t
        v = round(min(max(self.filter(self.normalise(raw), t), 0.0), 1.0) * 127)
        self.display = v
        return v

    def accept(self, v):
        """Hysteresis: drop 1-step reversals so a still hand doesn't flicker n, n+1, n, ..."""
        if v is None or v == self.value:
            return False
        if self.value is not None:
            step = v - self.value
            if abs(step) == 1 and step != self.direction and v not in (0, 127):
                return False
            self.direction = 1 if step > 0 else -1
        self.value = v
        return True

    def lost(self, t):
        """Hand gone: return the on_lost value once it's been missing a moment."""
        self.raw = None
        if t - self.last_seen < LOST_GRACE_S:
            return None
        self.filter.reset()
        if self.on_lost is None:
            return None
        self.display = int(self.on_lost)
        return self.display


def load_calibration():
    try:
        return json.loads(CALIB.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def open_midi(name):
    """Use an existing output whose name contains `name` (e.g. "IAC"), otherwise
    create a virtual port that shows up in Ableton as `name`."""
    for existing in mido.get_output_names():
        if name.lower() in existing.lower():
            return mido.open_output(existing), existing
    return mido.open_output(name, virtual=True), f"{name} (virtual)"


def ensure_camera_permission():
    """Ask macOS for camera access and wait for the answer. OpenCV requests it too,
    but gives up in the same instant, so the prompt never gets a chance to show."""
    try:
        import AVFoundation as AV
    except ImportError:
        return
    status = AV.AVCaptureDevice.authorizationStatusForMediaType_(AV.AVMediaTypeVideo)
    if status == AV.AVAuthorizationStatusAuthorized:
        return
    if status == AV.AVAuthorizationStatusNotDetermined:
        answer = []
        AV.AVCaptureDevice.requestAccessForMediaType_completionHandler_(AV.AVMediaTypeVideo, lambda granted: answer.append(granted))
        print("Waiting for you to allow camera access in the macOS prompt...")
        while not answer:
            time.sleep(0.1)
        if answer[0]:
            return
    sys.exit("Camera access is off for this terminal app. Turn it on in System Settings → Privacy & Security → "
             "Camera, or reset it with:  tccutil reset Camera com.apple.Terminal")


def open_camera(index, width, height):
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    ok = cap.isOpened() and cap.read()[0]
    if not ok:
        sys.exit(f"Couldn't read camera {index}. If macOS didn't ask for permission, enable your terminal app in\n"
                 "System Settings → Privacy & Security → Camera, then restart the terminal. "
                 "Try --list-cameras to find the right index.")
    return cap


def ensure_model():
    """Download the hand landmarker model next to this script if it isn't there yet."""
    if MODEL.exists():
        return
    print(f"Downloading {MODEL.name} (7.8 MB) from {MODEL_URL}")
    part = MODEL.with_name(MODEL.name + ".part")
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as r, open(part, "wb") as f:
            shutil.copyfileobj(r, f)
        part.rename(MODEL)
    except OSError as e:
        part.unlink(missing_ok=True)
        sys.exit(f"Couldn't download the model ({e}). Download it by hand with:\n"
                 f'  curl -L -o "{MODEL}" {MODEL_URL}')


def make_landmarker(mode, num_hands, cfg):
    ensure_model()
    return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL), delegate=BaseOptions.Delegate.CPU),  # GPU path crashes on macOS
        running_mode=mode,
        num_hands=num_hands,
        min_hand_detection_confidence=float(cfg.get("detection_confidence", 0.6)),
        min_hand_presence_confidence=float(cfg.get("presence_confidence", 0.5)),
        min_tracking_confidence=float(cfg.get("tracking_confidence", 0.5)),
    ))


def detected_hands(result, aspect, swap):
    """{"Left"/"Right": (image landmarks, features)} from a landmarker result."""
    hands = {}
    for lm, wlm, handed in zip(result.hand_landmarks, result.hand_world_landmarks, result.handedness):
        label = handed[0].category_name
        if swap:
            label = "Left" if label == "Right" else "Right"
        if label not in hands:
            hands[label] = (lm, hand_features(lm, wlm, aspect))
    return hands


# ---------------------------------------------------------------- drawing

HAND_COLOUR = {"Left": (255, 170, 60), "Right": (60, 200, 255)}


def draw_hands(frame, hands):
    h, w = frame.shape[:2]
    for label, (lm, _) in hands.items():
        pts = [(int(p.x * w), int(p.y * h)) for p in lm]
        colour = HAND_COLOUR[label]
        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], colour, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, p, 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(frame, label, (pts[WRIST][0] - 20, pts[WRIST][1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)


def draw_hud(frame, maps, solo, frozen, calibrating, fps, port_label):
    row, pad, width = 28, 12, 380
    height = pad * 2 + row * (len(maps) + 2)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    font = cv2.FONT_HERSHEY_SIMPLEX

    status = f"{port_label}  {fps:4.1f} fps"
    if frozen:
        status += "  FROZEN"
    if calibrating:
        status += "  CALIBRATING"
    cv2.putText(frame, status, (pad, pad + 16), font, 0.5,
                (80, 80, 255) if frozen or calibrating else (200, 200, 200), 1, cv2.LINE_AA)

    for i, m in enumerate(maps):
        y = pad + row * (i + 1) + 16
        muted = frozen or (solo is not None and solo != i)
        colour = (110, 110, 110) if muted else HAND_COLOUR.get(m.hand, (220, 220, 220))
        if solo == i:
            cv2.rectangle(frame, (4, y - 18), (width - 4, y + 8), (0, 180, 255), 1)
        cv2.putText(frame, f"{i + 1} {m.name[:16]}", (pad, y), font, 0.5, colour, 1, cv2.LINE_AA)
        cv2.putText(frame, f"CC{m.cc}", (180, y), font, 0.5, colour, 1, cv2.LINE_AA)
        x0, x1 = 250, width - 50
        cv2.rectangle(frame, (x0, y - 11), (x1, y + 1), (70, 70, 70), 1)
        if m.display is not None:
            fill = x0 + int((x1 - x0) * m.display / 127)
            cv2.rectangle(frame, (x0, y - 11), (fill, y + 1), colour, -1)
            cv2.putText(frame, f"{m.display:3d}", (x1 + 8, y), font, 0.5, colour, 1, cv2.LINE_AA)
        if m.raw is None:
            cv2.putText(frame, "-", (x0 - 14, y), font, 0.5, (110, 110, 110), 1, cv2.LINE_AA)

    keys = "1-9 solo  0 all  space freeze  c calibrate  x clear cal  a resend  q quit"
    cv2.putText(frame, keys, (pad, height - pad), font, 0.4, (160, 160, 160), 1, cv2.LINE_AA)


# ---------------------------------------------------------------- modes

def run_image(path, cfg):
    """Print features for a still image and save an annotated copy (for testing)."""
    frame = cv2.imread(str(path))
    if frame is None:
        sys.exit(f"Can't read {path}")
    with make_landmarker(vision.RunningMode.IMAGE, 2, cfg) as lm:
        result = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    hands = detected_hands(result, frame.shape[1] / frame.shape[0], cfg.get("swap_hands", False))
    for label, (_, feats) in hands.items():
        print(label, {k: round(v, 3) for k, v in feats.items()})
    if not hands:
        print("no hands found")
    draw_hands(frame, hands)
    out = Path(path).with_name(Path(path).stem + "_annotated.jpg")
    cv2.imwrite(str(out), frame)
    print("annotated →", out)


def list_cameras():
    found = False
    for i in range(6):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if cap.isOpened() and cap.read()[0]:
            found = True
            print(f"camera {i}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        cap.release()
    if not found:
        print("No readable cameras. Allow your terminal app in System Settings → Privacy & Security → Camera, "
              "then quit and reopen the terminal.")


def run_live(cfg, maps, camera_index):
    port, port_label = open_midi(cfg.get("port", "HandCC"))
    ensure_camera_permission()
    cap = open_camera(camera_index, int(cfg.get("width", 1280)), int(cfg.get("height", 720)))
    mirror = cfg.get("mirror", True)
    # The model labels hands correctly on the raw (unmirrored) camera image, so
    # flipping the frame for a selfie view flips its Left/Right too.
    swap = mirror != cfg.get("swap_hands", False)
    needs_two = {m.hand for m in maps} >= {"Left", "Right"}
    landmarker = make_landmarker(vision.RunningMode.VIDEO, 2 if needs_two else int(cfg.get("num_hands", 2)), cfg)

    solo, frozen, calibrating = None, False, False
    last_ts, fps, t_prev = -1, 0.0, time.monotonic()
    print(f"Sending on {port_label}. Focus the preview window for keys; q to quit.")

    def send(m, v):
        if m.accept(v):
            port.send(mido.Message("control_change", channel=m.channel, control=m.cc, value=v))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            if mirror:
                frame = cv2.flip(frame, 1)
            now = time.monotonic()
            ts = max(last_ts + 1, int(now * 1000))
            last_ts = ts
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts)
            hands = detected_hands(result, frame.shape[1] / frame.shape[0], swap)

            for i, m in enumerate(maps):
                if m.hand == "Any":
                    found = next(iter(hands.values()), None)
                else:
                    found = hands.get(m.hand)
                v = m.update(found[1][m.feature], now) if found else m.lost(now)
                active = solo is None or solo == i
                if calibrating and active and m.raw is not None:
                    m.samples.append(m.raw)
                if active and not frozen:
                    send(m, v)

            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3)
            t_prev = now
            draw_hands(frame, hands)
            draw_hud(frame, maps, solo, frozen, calibrating, fps, port_label)
            cv2.imshow("handcc", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif ord("1") <= key <= ord("9") and key - ord("1") < len(maps):
                solo = key - ord("1")
                print(f"solo: {maps[solo].name} (CC{maps[solo].cc})")
            elif key == ord("0"):
                solo = None
                print("solo off")
            elif key == ord(" "):
                frozen = not frozen
                print("frozen" if frozen else "unfrozen")
            elif key == ord("a"):
                for i, m in enumerate(maps):
                    if m.display is not None and (solo is None or solo == i):
                        m.value, m.direction = None, 0
                        send(m, m.display)
            elif key == ord("c"):
                if not calibrating:
                    for m in maps:
                        m.samples = []
                    calibrating = True
                    print("calibrating — move through the full range of each control, then press c")
                else:
                    calibrating = False
                    finish_calibration(maps)
            elif key == ord("x"):
                CALIB.unlink(missing_ok=True)
                for m in maps:
                    m.range = m.config_range
                print("calibration cleared")
    finally:
        landmarker.close()
        cap.release()
        port.close()
        cv2.destroyAllWindows()


def finish_calibration(maps):
    calibration = load_calibration()
    for m in maps:
        if len(m.samples) < 15:
            continue
        # Percentiles, not min/max, so one glitchy frame can't stretch the range.
        lo, hi = np.percentile(m.samples, [3, 97])
        default_lo, default_hi = DEFAULT_RANGE[m.feature]
        if hi - lo < 0.2 * (default_hi - default_lo):
            # Barely moved: a range this narrow would make the CC jump 0→127 on a twitch.
            print(f"  {m.name}: not enough movement, kept {m.range[0]:.3f} … {m.range[1]:.3f}")
            m.samples = []
            continue
        m.range = (float(lo), float(hi))
        calibration[m.name] = list(m.range)
        print(f"  {m.name}: {lo:.3f} … {hi:.3f}")
        m.samples = []
    CALIB.write_text(json.dumps(calibration, indent=2))
    print(f"calibration saved → {CALIB.name}")


def main():
    ap = argparse.ArgumentParser(description="Webcam hand tracking → MIDI CC")
    ap.add_argument("--config", default=HERE / "mappings.toml", type=Path)
    ap.add_argument("--camera", type=int, help="camera index (overrides config)")
    ap.add_argument("--list-cameras", action="store_true")
    ap.add_argument("--list-ports", action="store_true", help="list MIDI outputs")
    ap.add_argument("--image", type=Path, help="analyse a still image instead of the camera")
    args = ap.parse_args()

    if args.list_ports:
        print("\n".join(mido.get_output_names()) or "(no MIDI outputs)")
        return
    if args.list_cameras:
        ensure_camera_permission()
        list_cameras()
        return

    cfg = tomllib.loads(args.config.read_text())
    if args.image:
        run_image(args.image, cfg)
        return

    calibration = load_calibration()
    maps = [Mapping(spec, cfg, calibration) for spec in cfg.get("map", [])]
    if not maps:
        sys.exit(f"No [[map]] entries in {args.config}")
    if len({m.name for m in maps}) != len(maps):
        sys.exit("Mapping names must be unique (calibration is stored by name)")
    run_live(cfg, maps, args.camera if args.camera is not None else int(cfg.get("camera", 0)))


if __name__ == "__main__":
    main()
