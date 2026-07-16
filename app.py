"""
Body Measurement Studio — Streamlit app
---------------------------------------
Upload a full-body photo (+ optional side photo), enter height & weight,
get 14 body measurements with an honest confidence read-out.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import io
import os
os.environ["GLOG_minloglevel"] = "3"  # suppress MediaPipe verbose logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import cv2
import streamlit as st
import pandas as pd

from measure_engine import MeasurementEngine, _MP_OK, _MP_MODE, _MP_ERROR
from smpl_engine import SMPLMeasurementEngine
from size_recommender import (recommend_sizes, format_size_card,
                              format_size_summary, get_available_brands,
                              get_fit_types)

st.set_page_config(page_title="Body Measurement Studio",
                   page_icon="📐", layout="wide")

# --------------------------- styling -------------------------------------- #
st.markdown("""
<style>
    .metric-card {background:#12151c;border:1px solid #232a36;border-radius:12px;
        padding:14px 16px;margin-bottom:10px;}
    .metric-name {color:#8b96a8;font-size:13px;text-transform:uppercase;
        letter-spacing:.06em;}
    .metric-val {color:#e8edf5;font-size:26px;font-weight:600;}
    .metric-unit {color:#5f6b7d;font-size:14px;font-weight:400;}
    .note {background:#1a1408;border-left:3px solid #b8860b;padding:8px 12px;
        border-radius:6px;margin:6px 0;color:#d8c9a0;font-size:13px;}
</style>
""", unsafe_allow_html=True)

st.title("📐 Body Measurement Studio")
st.caption("Single-photo anthropometry, calibrated by your real height & weight. "
           "Baseline geometry engine — add a side photo for best accuracy.")


@st.cache_resource
def get_engine():
    engine = SMPLMeasurementEngine()
    if engine.is_smpl_available:
        return engine
    return MeasurementEngine()


def show_setup_status():
    """Pose detection is load-bearing: without it the app cannot separate arms
    from the torso and every girth falls back to a population model. So surface
    its status up front rather than letting it fail silently mid-result."""
    if _MP_OK and _MP_ERROR is None:
        st.success(f"Pose detection ready (MediaPipe {_MP_MODE} API). "
                   "Girths will be measured from your photo.")
    else:
        st.error(
            "**Pose detection unavailable — girths will fall back to a "
            "height/BMI population model, not your actual body.**\n\n"
            f"{_MP_ERROR or 'MediaPipe not installed.'}\n\n"
            "Fix: `pip install mediapipe` (or pin `mediapipe==0.10.14`). "
            "On first run the app downloads a ~30 MB pose model.")


def read_upload(file) -> np.ndarray:
    data = np.frombuffer(file.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# --------------------------- sidebar inputs ------------------------------- #
with st.sidebar:
    st.header("Your details")
    unit = st.radio("Height/weight units", ["Metric (cm / kg)",
                                            "Imperial (ft-in / lb)"],
                    horizontal=False)
    if unit.startswith("Metric"):
        height = st.number_input("Height (cm)", 120.0, 220.0, 170.0, 0.5)
        weight = st.number_input("Weight (kg)", 30.0, 200.0, 68.0, 0.5)
        height_cm, weight_kg = height, weight
        # Show converted
        ft = int(height_cm // 30.48)
        inch = (height_cm - ft * 30.48) / 2.54
        st.caption(f"= {ft}' {inch:.1f}\" / {weight_kg * 2.205:.1f} lb")
    else:
        hcol1, hcol2 = st.columns(2)
        with hcol1:
            height_ft = st.number_input("Feet", 3, 7, 5, 1)
        with hcol2:
            height_in = st.number_input("Inches", 0.0, 11.9, 7.0, 0.5)
        weight_lb = st.number_input("Weight (lb)", 66.0, 440.0, 150.0, 1.0)
        height_cm = (height_ft * 12 + height_in) * 2.54
        weight_kg = weight_lb * 0.4536
        # Show converted
        st.caption(f"= {height_cm:.1f} cm / {weight_kg:.1f} kg")

    display_unit = st.radio("Show measurements in", ["cm", "inches", "both"],
                            horizontal=True)

    sex = st.selectbox("Gender",
                       ["neutral", "male", "female"])
    st.divider()
    st.header("Size preferences")
    fit_type = st.selectbox("Fit type", get_fit_types(), index=1)
    available_brands = get_available_brands()
    selected_brands = st.multiselect(
        "Brands to check", available_brands,
        default=available_brands,
        help="Select which brands to get size recommendations for")
    st.divider()
    st.markdown("**For best results**")
    st.markdown("- Full body in frame, standing straight\n"
                "- Tight/fitted clothing, plain background\n"
                "- Phone at chest height, ~2–3 m away\n"
                "- Arms slightly away from body (A-pose)")


# --------------------------- image inputs --------------------------------- #
c1, c2 = st.columns(2)
with c1:
    st.subheader("Front photo (required)")
    front_file = st.file_uploader("Front view", type=["jpg", "jpeg", "png"],
                                  key="front")
with c2:
    st.subheader("Side photo (optional — big accuracy boost)")
    side_file = st.file_uploader("Side view", type=["jpg", "jpeg", "png"],
                                 key="side")

show_setup_status()

run = st.button("Measure", type="primary", width="stretch",
                disabled=front_file is None)

if run and front_file is not None:
    front_bgr = read_upload(front_file)
    side_bgr = read_upload(side_file) if side_file is not None else None

    if front_bgr is None:
        st.error("Couldn't read the front image. Try a JPG or PNG.")
        st.stop()

    with st.spinner("Detecting pose, segmenting silhouette, computing "
                    "measurements..."):
        engine = get_engine()
        result = engine.measure(front_bgr, height_cm, weight_kg, sex,
                                side_bgr=side_bgr)

    # ---- confidence + photo-trust banner ---- #
    conf_pct = int(result.confidence * 100)
    trust_pct = int(result.photo_trust * 100)
    color = "#2e7d32" if conf_pct >= 70 else ("#b8860b" if conf_pct >= 50
                                              else "#c0392b")
    b1, b2 = st.columns(2)
    with b1:
        st.markdown(
            f"<div style='background:{color}22;border:1px solid {color};"
            f"border-radius:10px;padding:12px 16px;'>"
            f"<b style='color:{color};'>Confidence: {conf_pct}%</b>"
            f"{' · side view used' if result.used_side_view else ''}</div>",
            unsafe_allow_html=True)
    with b2:
        tcol = "#2e7d32" if trust_pct >= 60 else ("#b8860b" if trust_pct >= 30
                                                  else "#c0392b")
        st.markdown(
            f"<div style='background:{tcol}22;border:1px solid {tcol};"
            f"border-radius:10px;padding:12px 16px;'>"
            f"<b style='color:{tcol};'>Photo trust: {trust_pct}%</b> "
            f"<span style='color:#8b96a8;font-size:12px;'>how much of the "
            f"result came from your image vs. the height/BMI model</span></div>",
            unsafe_allow_html=True)

    # Warnings first — these are the things that make numbers wrong.
    for w in result.warnings:
        st.error(w)

    for n in result.notes:
        st.markdown(f"<div class='note'>{n}</div>", unsafe_allow_html=True)

    # ---- visuals ---- #
    v1, v2 = st.columns(2)
    with v1:
        st.markdown("**Detected landmarks & measurement lines**")
        st.image(cv2.cvtColor(result.landmarks_img, cv2.COLOR_BGR2RGB),
                 width="stretch")
    with v2:
        st.markdown("**Silhouette**")
        st.image(cv2.cvtColor(result.silhouette_img, cv2.COLOR_BGR2RGB),
                 width="stretch")

    # ---- measurements ---- #
    st.subheader("Measurements")
    linear = {k: v for k, v in result.measurements.items()
              if k not in ("BMI", "Weight")}
    BADGE = {
        "measured": ("#2e7d32", "from photo"),
        "blended":  ("#b8860b", "photo + model"),
        "modeled":  ("#c0392b", "model only"),
        "given":    ("#5f6b7d", "you entered"),
    }

    def _format_val(name, val_cm):
        """Format measurement value based on selected display unit."""
        if name == "Height":
            if display_unit == "inches":
                total_in = val_cm / 2.54
                ft = int(total_in // 12)
                inch = total_in - ft * 12
                return f"{ft}' {inch:.1f}\"", ""
            elif display_unit == "both":
                total_in = val_cm / 2.54
                ft = int(total_in // 12)
                inch = total_in - ft * 12
                return f"{val_cm}", f"cm ({ft}' {inch:.1f}\")"
            return f"{val_cm}", "cm"
        if display_unit == "inches":
            return f"{val_cm / 2.54:.1f}", "in"
        elif display_unit == "both":
            return f"{val_cm}", f"cm ({val_cm / 2.54:.1f} in)"
        return f"{val_cm}", "cm"

    cols = st.columns(3)
    for i, (name, val) in enumerate(linear.items()):
        val_str, unit_lbl = _format_val(name, val)
        src_kind = result.provenance.get(name, "measured")
        bc, blabel = BADGE.get(src_kind, BADGE["measured"])
        with cols[i % 3]:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-name'>{name}</div>"
                f"<div class='metric-val'>{val_str} "
                f"<span class='metric-unit'>{unit_lbl}</span></div>"
                f"<div style='margin-top:6px;'>"
                f"<span style='background:{bc}22;color:{bc};border:1px solid "
                f"{bc}55;border-radius:20px;padding:2px 9px;font-size:11px;"
                f"font-weight:600;'>{blabel}</span></div>"
                f"</div>",
                unsafe_allow_html=True)

    n_modeled = sum(1 for k in linear if
                    result.provenance.get(k) in ("modeled", "blended"))
    if n_modeled:
        st.warning(
            f"**{n_modeled} of {len(linear)} values are not measurements of "
            f"this person.** Anything tagged *model only* is a typical value "
            f"for someone {height_cm:.0f} cm / {weight_kg:.0f} kg — it would "
            f"read the same for any body of that size. Only *from photo* "
            f"values are specific to the person in the image.")

    if "BMI" in result.measurements:
        st.info(f"BMI: **{result.measurements['BMI']}**  "
                f"(from your entered height & weight)")

    # ---- export ---- #
    df = pd.DataFrame(
        [{"measurement": k, "value_cm": v,
          "source": result.provenance.get(k, "measured")}
         for k, v in result.measurements.items()])
    st.download_button("Download measurements (CSV)",
                       df.to_csv(index=False).encode(),
                       file_name="measurements.csv", mime="text/csv")

    # ---- size recommendations ---- #
    st.subheader("👕 Suggested Sizes")
    size_recs = recommend_sizes(result.measurements, sex, fit_type,
                                selected_brands or None)

    if size_recs:
        summary = format_size_summary(size_recs)
        if summary:
            st.markdown(f"<div style='background:#0d1117;border:1px solid "
                        f"#30363d;border-radius:10px;padding:12px 16px;"
                        f"margin-bottom:16px;font-size:15px;'>"
                        f"🎯 {summary}</div>", unsafe_allow_html=True)

        # Group by garment type
        tops = [r for r in size_recs if r.garment == "Top"]
        bottoms = [r for r in size_recs if r.garment == "Bottom"]

        if tops:
            st.markdown("**Tops (T-shirt / Shirt / Jacket)**")
            tcols = st.columns(min(len(tops), 4))
            for i, rec in enumerate(tops):
                with tcols[i % len(tcols)]:
                    st.markdown(format_size_card(rec), unsafe_allow_html=True)

        if bottoms:
            st.markdown("**Bottoms (Pants / Jeans)**")
            bcols = st.columns(min(len(bottoms), 4))
            for i, rec in enumerate(bottoms):
                with bcols[i % len(bcols)]:
                    st.markdown(format_size_card(rec), unsafe_allow_html=True)
    else:
        st.info("No size recommendations available. Select brands in the sidebar.")

    with st.expander("ℹ️ How size scoring works"):
        st.markdown(
            "Each size is scored by comparing **all your measurements** "
            "(chest, waist, hip, shoulder, height, inseam) against the brand's "
            "size chart simultaneously. Measurements are weighted by importance "
            "(e.g. chest matters most for tops, waist for bottoms).\n\n"
            "- **Fit score 80+** = measurements land in the sweet spot\n"
            "- **Fit score 65–80** = fits but some areas may be snug/loose\n"
            "- **Fit score <65** = likely uncomfortable, try adjacent size\n\n"
            "The colored pills show which specific measurements are tight/loose "
            "in that size. Add your own brands by editing `size_charts.json`.")


    with st.expander("⚠️ How accurate is this? (read before trusting numbers)",
                     expanded=trust_pct < 40):
        st.markdown(
            "- **Height** is exactly what you entered — it's the scale anchor, "
            "not a prediction.\n"
            "- **Widths** (shoulder, and the width component of girths) come "
            "from the silhouette and are fairly reliable.\n"
            "- **Girths** (chest/waist/hip) need depth. Front-only infers depth "
            "from BMI → expect **±3–6 cm**. With a side photo depth is measured "
            "→ **±2–3 cm**.\n"
            "- Loose clothing, tilted camera, cropped limbs, or a busy "
            "background degrade everything. This is a baseline demo, not a "
            "medical or tailoring-grade tool.\n"
            "- To reach ±1.5 cm you'd train a BMnet-style regressor on the "
            "BodyM dataset (front+side silhouettes + height + weight) and drop "
            "it into `MeasurementEngine` — the app interface won't change.\n\n"
            "**Photo trust** tells you how much of the girth numbers actually "
            "came from your image. When it's low, the silhouette was "
            "untrustworthy (loose clothing, arms against the body, graphics in "
            "the frame) and the app fell back on a height/BMI population model. "
            "Those numbers are then *typical values for your size*, not "
            "measurements of you. Re-shoot in fitted clothing, A-pose, plain "
            "background to raise it.")

elif not run:
    st.info("Upload a front photo, enter your height & weight, then hit "
            "**Measure**. A side photo is optional but roughly halves girth "
            "error.")