"""
gesture/trainer.py — Optional scaffold for collecting custom gesture training data.

This module lets you record labelled hand-landmark snapshots to a JSON file.
You can later use these to train a small sklearn or TFLite classifier and
replace the heuristic logic in recognizer.py.

Usage (command-line)
--------------------
  python -m gesture.trainer --label gojo --count 200 --out data/gestures.json
"""

import json
import os
import argparse
import time
from typing import List

import cv2

from camera.webcam import Webcam
from gesture.detector import HandDetector


def collect_samples(
    label:      str,
    count:      int,
    out_path:   str,
    delay_sec:  float = 0.05
) -> None:
    """
    Record `count` hand-landmark samples labelled as `label` and append
    them to `out_path` (JSON Lines format).

    Parameters
    ----------
    label     : Gesture label (e.g. "gojo", "sukuna", "neutral")
    count     : Number of frames to capture
    out_path  : Output .json or .jsonl file path
    delay_sec : Seconds to wait between captures
    """
    cam      = Webcam()
    detector = HandDetector()

    if not cam.open():
        print("ERROR: Could not open webcam.")
        return

    detector.start()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    samples: List[dict] = []
    print(f"\n[Trainer] Collecting {count} samples for label='{label}'.")
    print("  Show your gesture to the camera. Press ESC to abort.\n")

    # Countdown
    for i in range(3, 0, -1):
        print(f"  Starting in {i}...")
        time.sleep(1)

    captured = 0
    while captured < count:
        frame = cam.get_frame()
        if frame is None:
            continue

        hands = detector.detect(frame)

        if hands:
            sample = {
                "label":     label,
                "landmarks": hands[0].landmarks,   # normalised (x,y,z) × 21
            }
            samples.append(sample)
            captured += 1
            print(f"  [{captured}/{count}] captured", end="\r")

        # Display live preview
        cv2.putText(
            frame,
            f"Collecting '{label}': {captured}/{count}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        cv2.imshow("Trainer", frame)

        if cv2.waitKey(1) & 0xFF == 27:   # ESC
            print("\n  Aborted by user.")
            break

        time.sleep(delay_sec)

    # Save
    with open(out_path, "a") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"\n[Trainer] Saved {len(samples)} samples → {out_path}")

    detector.stop()
    cam.release()
    cv2.destroyAllWindows()


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JJK Gesture Trainer")
    parser.add_argument("--label", required=True,  help="Gesture label to record")
    parser.add_argument("--count", type=int, default=200, help="Number of samples")
    parser.add_argument("--out",   default="data/gestures.jsonl", help="Output file")
    args = parser.parse_args()

    collect_samples(args.label, args.count, args.out)
