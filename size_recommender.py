"""
size_recommender.py — Garment-formula based size recommendation
---------------------------------------------------------------
Uses standard garment sizing logic:
  1. Body measurement + ease allowance = required garment measurement
  2. Match garment measurement against brand size charts
  3. Height adjusts for sleeve/body length — tall people size up

Primary sizing dimensions:
  - Tops: Chest (primary) + Neck (collar) + Shoulder + Sleeve + Height
  - Bottoms: Waist (primary) + Hip + Inseam
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Optional

_CHART_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "size_charts.json")
_CHARTS = None


def _load_charts() -> dict:
    global _CHARTS
    if _CHARTS is not None:
        return _CHARTS
    if os.path.exists(_CHART_PATH):
        with open(_CHART_PATH, "r") as f:
            _CHARTS = json.load(f)
    else:
        _CHARTS = {"brands": {}, "measurement_weights": {}, "fit_types": {}}
    return _CHARTS


@dataclass
class SizeRecommendation:
    garment: str
    size: str
    brand: str
    fit_score: float
    fit_note: str
    size_system: str = ""
    numeric_size: Optional[str] = None
    per_measurement: dict = None


# ---------- Garment sizing formulas --------------------------------------- #

# Ease allowances (cm added to body measurement for garment comfort)
_EASE = {
    "slim":    {"chest": 8, "waist": 6, "shoulder": 0.5, "sleeve": 0, "length": 0},
    "regular": {"chest": 10, "waist": 8, "shoulder": 1.0, "sleeve": 1, "length": 0},
    "relaxed": {"chest": 14, "waist": 12, "shoulder": 1.5, "sleeve": 2, "length": 2},
    "oversized": {"chest": 18, "waist": 16, "shoulder": 2.5, "sleeve": 2, "length": 4},
}


def _compute_garment_needs(measurements: dict, fit_type: str = "regular") -> dict:
    """Convert body measurements to required garment measurements using ease."""
    ease = _EASE.get(fit_type, _EASE["regular"])

    chest = measurements.get("Chest / bust", 0)
    waist = measurements.get("Waist", 0)
    shoulder = measurements.get("Shoulder width", 0)
    neck = measurements.get("Neck", 0)
    sleeve = measurements.get("Arm length (sh\u2192wrist)", 0)
    height = measurements.get("Height", 170)
    hip = measurements.get("Hip", 0)
    inseam = measurements.get("Inseam", 0)

    # Shirt length from height (empirical: ~0.42 * height for regular)
    shirt_length = height * 0.42 + ease["length"]

    # Collar = neck circumference (no ease needed, already measured around)
    collar_cm = neck
    collar_in = round(neck / 2.54, 1) if neck > 0 else 0

    return {
        "garment_chest": chest + ease["chest"] if chest > 0 else 0,
        "garment_shoulder": shoulder + ease["shoulder"] if shoulder > 0 else 0,
        "garment_sleeve": sleeve + ease["sleeve"] if sleeve > 0 else 0,
        "garment_length": shirt_length,
        "collar_cm": collar_cm,
        "collar_in": collar_in,
        "body_chest": chest,
        "body_waist": waist,
        "body_hip": hip,
        "body_inseam": inseam,
        "height": height,
    }


def _find_best_size(body_value: float, size_chart: dict, key: str) -> tuple:
    """Find the best matching size for a body measurement value.

    Returns (size_label, score, fit_detail).
    """
    best_size = None
    best_score = -1
    best_detail = ""

    for size_label, chart in size_chart.items():
        if key not in chart:
            continue
        lo, hi = chart[key]
        if lo == hi:
            hi = lo + 4  # single-value tolerance

        rng = max(1, hi - lo)

        if lo <= body_value <= hi:
            position = (body_value - lo) / rng
            if position > 0.85:
                score = 88.0
                detail = "slightly tight"
            elif position < 0.15:
                score = 90.0
                detail = "slightly loose"
            else:
                score = 95.0
                detail = "perfect"
        elif body_value < lo:
            gap = (lo - body_value) / rng
            score = max(40, 85.0 - 25.0 * gap)
            detail = "loose"
        else:
            gap = (body_value - hi) / rng
            score = max(20, 75.0 - 40.0 * gap)
            detail = "tight"

        if score > best_score:
            best_score = score
            best_size = size_label
            best_detail = detail

    return best_size, best_score, best_detail


# ---------- Height-based size adjustment ---------------------------------- #

def _height_size_adjustment(height: float, base_size: str, sizes_list: list) -> str:
    """Adjust size up if height requires longer garment.

    At 183cm+, a shirt that fits the chest may be too short in body/sleeves.
    Standard rule: if height > 180cm and current size is based on chest alone,
    consider sizing up.
    """
    if height <= 180 or not sizes_list:
        return base_size

    try:
        idx = sizes_list.index(base_size)
    except ValueError:
        return base_size

    # Height > 180: if not already at L or above, consider size up
    # Height > 185: definitely size up unless already XL+
    large_sizes = {"L", "XL", "XXL", "2XL", "3XL"}
    if base_size in large_sizes:
        return base_size

    if height >= 185 and idx + 1 < len(sizes_list):
        return sizes_list[idx + 1]
    if height >= 180 and idx + 1 < len(sizes_list):
        # Only size up if the next size isn't a huge jump
        return sizes_list[idx + 1]

    return base_size


# ---------- Public API ---------------------------------------------------- #

# Alpha sizes in order — used for consensus voting
_ALPHA_ORDER = ["XXS", "XS", "S", "M", "L", "XL", "XXL", "2XL", "3XL"]


def _is_alpha(size: str) -> bool:
    return size in _ALPHA_ORDER


def recommend_sizes(measurements: dict, sex: str = "neutral",
                    fit_type: str = "regular",
                    brands: list = None) -> list[SizeRecommendation]:
    """Generate ONE size recommendation per garment type.

    Scores all brands internally, takes a consensus (majority vote on alpha
    sizes), and returns a single Top and single Bottom recommendation.
    """
    charts = _load_charts()
    all_brands = charts.get("brands", {})

    if brands:
        all_brands = {k: v for k, v in all_brands.items() if k in brands}

    gender = "women" if sex == "female" else "men"
    needs = _compute_garment_needs(measurements, fit_type)

    top_votes = []    # list of alpha sizes from each brand
    bot_votes = []    # list of alpha sizes from each brand
    top_scores = {}   # size -> list of scores
    bot_scores = {}   # size -> list of scores

    for brand_name, brand_data in all_brands.items():
        # --- Tops ---
        tops = brand_data.get("tops", {}).get(gender, {})
        if tops and needs["body_chest"] > 0:
            size, score, detail = _find_best_size(needs["body_chest"], tops, "chest")
            if size:
                sizes_list = list(tops.keys())
                # If at the very top of this size's range, vote for next size up
                if detail == "slightly tight" and size in tops:
                    try:
                        idx = sizes_list.index(size)
                        if idx + 1 < len(sizes_list):
                            size = sizes_list[idx + 1]
                    except ValueError:
                        pass
                size = _height_size_adjustment(needs["height"], size, sizes_list)
                if _is_alpha(size):
                    top_votes.append(size)
                    top_scores.setdefault(size, []).append(score)

        # --- Bottoms ---
        bottoms = brand_data.get("bottoms", {}).get(gender, {})
        if bottoms and needs["body_waist"] > 0:
            size, score, detail = _find_best_size(needs["body_waist"], bottoms, "waist")
            if size and _is_alpha(size):
                bot_votes.append(size)
                bot_scores.setdefault(size, []).append(score)

    results = []

    # --- Consensus Top ---
    if top_votes:
        from collections import Counter
        top_size = Counter(top_votes).most_common(1)[0][0]
        avg_score = sum(top_scores[top_size]) / len(top_scores[top_size])
        # Re-derive fit detail for the consensus size
        chest = needs["body_chest"]
        # Find this size in any brand chart to get range
        detail = _consensus_detail(chest, top_size, all_brands, gender, "tops", "chest")
        per_meas = {"Chest": detail}
        if needs["collar_cm"] > 0:
            per_meas["Collar"] = f"{round(needs['collar_cm'])} cm ({needs['collar_in']:.0f}\")"
        note = _build_fit_note(avg_score, {"Chest": detail}, fit_type)
        if needs["height"] > 180 and top_size not in {"L", "XL", "XXL"}:
            note = f"Sized up for height ({needs['height']:.0f} cm)"
        results.append(SizeRecommendation(
            garment="Top", size=top_size, brand="",
            fit_score=round(avg_score, 1), fit_note=note,
            per_measurement=per_meas))

    # --- Consensus Bottom ---
    if bot_votes:
        from collections import Counter
        bot_size = Counter(bot_votes).most_common(1)[0][0]
        avg_score = sum(bot_scores[bot_size]) / len(bot_scores[bot_size])
        detail = _consensus_detail(needs["body_waist"], bot_size, all_brands, gender, "bottoms", "waist")
        per_meas = {"Waist": detail}
        note = _build_fit_note(avg_score, per_meas, fit_type)
        results.append(SizeRecommendation(
            garment="Bottom", size=bot_size, brand="",
            fit_score=round(avg_score, 1), fit_note=note,
            per_measurement=per_meas))

    return results


def _consensus_detail(value: float, size: str, all_brands: dict,
                      gender: str, garment_key: str, meas_key: str) -> str:
    """Get fit detail for a consensus size by checking it against any brand chart."""
    for brand_data in all_brands.values():
        chart = brand_data.get(garment_key, {}).get(gender, {})
        if size in chart and meas_key in chart[size]:
            lo, hi = chart[size][meas_key]
            if lo == hi:
                hi = lo + 4
            rng = max(1, hi - lo)
            if lo <= value <= hi:
                pos = (value - lo) / rng
                if pos > 0.85:
                    return "slightly tight"
                elif pos < 0.15:
                    return "slightly loose"
                return "perfect"
            elif value < lo:
                return "loose"
            else:
                return "tight"
    return "perfect"


def _build_fit_note(score: float, details: dict, fit_type: str) -> str:
    """Build human-readable fit note."""
    tight = [k for k, v in details.items() if "tight" in str(v)]
    loose = [k for k, v in details.items() if "loose" in str(v)]

    if score >= 85:
        base = "Excellent fit"
    elif score >= 70:
        base = "Good fit"
    elif score >= 55:
        base = "Acceptable fit"
    else:
        base = "Poor fit \u2014 consider adjacent size"

    parts = [base]
    if tight:
        parts.append(f"tight on {', '.join(tight)}")
    if loose:
        parts.append(f"loose on {', '.join(loose)}")

    return " \u00b7 ".join(parts)


def get_available_brands() -> list[str]:
    charts = _load_charts()
    return list(charts.get("brands", {}).keys())


def get_fit_types() -> list[str]:
    charts = _load_charts()
    return list(charts.get("fit_types", {}).keys())


# ---------- HTML formatting ----------------------------------------------- #

def format_size_card(rec: SizeRecommendation) -> str:
    if rec.fit_score >= 80:
        color = "#2e7d32"
        badge = "\u2713 Great fit"
    elif rec.fit_score >= 65:
        color = "#b8860b"
        badge = "~ Okay fit"
    else:
        color = "#c0392b"
        badge = "\u2717 Check fit"

    details_html = ""
    if rec.per_measurement:
        pills = []
        for meas, fit in rec.per_measurement.items():
            if "perfect" in str(fit):
                pc = "#2e7d32"
            elif "tight" in str(fit):
                pc = "#c0392b"
            elif "loose" in str(fit):
                pc = "#b8860b"
            else:
                pc = "#5f6b7d"
            pills.append(
                f"<span style='background:{pc}18;color:{pc};border:1px solid "
                f"{pc}44;border-radius:12px;padding:2px 7px;font-size:10px;"
                f"margin:2px;display:inline-block;'>{meas}: {fit}</span>"
            )
        details_html = f"<div style='margin-top:8px;'>{''.join(pills)}</div>"

    return (
        f"<div style='background:#0d1117;border:1px solid #30363d;"
        f"border-radius:10px;padding:14px 16px;margin-bottom:10px;'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
        f"<span style='color:#8b96a8;font-size:12px;text-transform:uppercase;"
        f"letter-spacing:.05em;'>{rec.garment}</span>"
        f"<span style='background:{color}22;color:{color};border:1px solid "
        f"{color}55;border-radius:12px;padding:2px 8px;font-size:11px;"
        f"font-weight:600;'>{badge} {rec.fit_score:.0f}%</span></div>"
        f"<div style='font-size:36px;font-weight:700;color:#58a6ff;"
        f"margin:6px 0;'>{rec.size}</div>"
        f"<div style='color:#a0aab8;font-size:12px;'>{rec.fit_note}</div>"
        f"{details_html}"
        f"</div>"
    )


def format_size_summary(results: list[SizeRecommendation]) -> str:
    from collections import Counter
    tops = [r.size for r in results if r.garment == "Top"]
    bottoms = [r.size for r in results if r.garment == "Bottom"]

    summary_parts = []
    if tops:
        most_common_top = Counter(tops).most_common(1)[0][0]
        summary_parts.append(f"**Tops: {most_common_top}**")
    if bottoms:
        most_common_bot = Counter(bottoms).most_common(1)[0][0]
        summary_parts.append(f"**Bottoms: {most_common_bot}**")

    return " \u00b7 ".join(summary_parts) if summary_parts else ""
