# Body Measurement Studio

Estimate body measurements from a single photo (front, or front+side),
**calibrated by your real height and weight**.

## What this is
A runnable baseline that needs **no model training**:

```
Photo ──▶ MediaPipe Pose (landmarks) ──┐
     └──▶ rembg/GrabCut (silhouette) ──┤──▶ height locks scale (cm/px)
                                        │──▶ widths at chest/waist/hip
     weight ──▶ BMI ──▶ depth ratios ───┘──▶ elliptical girths ──▶ 14 measures
     (or SIDE photo ──▶ measured depths, ~2x more accurate)
```

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
First run downloads the rembg model (~170 MB) once.

## Outputs (14)
Height, weight, shoulder width, chest/bust, waist, hip, neck, arm length,
thigh, inseam, outseam, torso length, knee height, BMI. Export to CSV.

## Accuracy — read this
| Setup | Girth error (chest/waist/hip) |
|---|---|
| Front only (depth inferred from BMI) | ±3–6 cm |
| Front + side (depth measured) | ±2–3 cm |

Height is exact (it's the scale anchor, not a prediction). Widths are reliable;
depth is the hard part from a single frontal view, which is why the side photo
matters so much. Loose clothing / tilted camera / cropped limbs degrade results.

## Upgrading to tailoring-grade (±1.5 cm)
The app talks to `MeasurementEngine.measure(...)` through a **stable interface**.
To upgrade without touching the UI:

1. Train a **BMnet-style regressor** on the **BodyM** dataset
   (front+side silhouettes + height + weight → 14 measurements).
2. Subclass `MeasurementEngine`, override `measure()` to run your network,
   return the same `MeasurementResult`.
3. Swap `get_engine()` in `app.py` to your subclass. Done.

For a 3D avatar (virtual try-on): route to **SMPL-X + SHAPY**, scale the mesh to
the known height, optimise shape betas so mesh-volume×density matches weight,
then read circumferences with **SMPL-Anthropometry**.

## Files
- `app.py` — Streamlit UI
- `measure_engine.py` — CV + geometry engine (the swappable part)
- `requirements.txt`
