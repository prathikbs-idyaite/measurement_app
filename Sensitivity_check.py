"""
sensitivity_check.py
--------------------
The decisive test for whether a "measurement" is real.

Hold the PHOTO fixed. Change only the CLAIMED weight. Any value that moves was
never measured from the image -- it was predicted from height+weight, and would
read the same for anyone of that size.

Run:  python sensitivity_check.py path/to/photo.jpg 170
"""
import sys
import cv2
from measure_engine import MeasurementEngine

def main(path, height_cm=170.0):
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"Could not read {path}")
    eng = MeasurementEngine()
    lo = eng.measure(img, height_cm, 60.0, "male")
    hi = eng.measure(img, height_cm, 90.0, "male")

    print(f"\nPhoto: {path}   height fixed at {height_cm:.0f} cm")
    print(f"Photo trust: {lo.photo_trust*100:.0f}%\n")
    print(f"{'measurement':24s}{'@60kg':>8s}{'@90kg':>8s}{'delta':>8s}  verdict")
    print("-" * 68)

    real, fake = [], []
    for k in lo.measurements:
        if k in ("Height", "Weight", "BMI"):
            continue
        a, b = lo.measurements[k], hi.measurements[k]
        d = b - a
        if abs(d) > 1.5:
            verdict = "PREDICTED from weight -- not measured"
            fake.append(k)
        else:
            verdict = "measured from the image"
            real.append(k)
        print(f"{k:24s}{a:>8}{b:>8}{d:>+8.1f}  {verdict}")

    print(f"\n{len(real)} genuinely measured: {', '.join(real)}")
    print(f"{len(fake)} predicted, not measured: {', '.join(fake)}")
    if fake:
        print("\nThe predicted values would read the same for ANY body of this "
              "height and weight. To turn them into real measurements: fitted "
              "clothing, A-pose (arms 30-45 deg out), plain background, plus a "
              "side photo.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 170.0)