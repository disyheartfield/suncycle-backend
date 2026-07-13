"""
visualization.py — Terminal output, GeoJSON export, Folium map
==============================================================

Three output modes:

1. TERMINAL: Colour-coded segment-by-segment output using colorama.
   Yellow/orange = sunny segments, blue/grey = shaded segments.
   Gives immediate feedback without needing a browser.

2. GEOJSON: Standard geographic format readable by QGIS, Mapbox, Leaflet,
   kepler.gl, etc. Each segment gets a `sun_score` property so it can be

   styled by any GIS tool. Useful for debugging and sharing.

3. FOLIUM MAP: Interactive HTML map with colour-coded route lines.
   Yellow = sunny, blue = shaded. Opens in any browser.
   The fastest way to visually verify the scoring is working correctly.

WHY SEGMENT-BASED COLOURING?
Rather than colouring individual points (which would look like a dotted line),
we colour runs of consecutive sun/shade points as continuous polylines.
This matches how the user will experience the route — as stretches, not points.
"""

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

from colorama import Fore, Back, Style, init

from scoring import RouteScore, PointScore, RankedRoutes
from routing import RouteCandidate

init(autoreset=True)


# ── Colour helpers ─────────────────────────────────────────────────────────────

def _sun_colour_terminal(score: float, in_sun: bool) -> str:
    """Map sun score to terminal colour."""
    if not in_sun:
        return Fore.BLUE + Style.DIM
    if score > 0.7:
        return Fore.YELLOW
    if score > 0.4:
        return Fore.WHITE
    return Fore.CYAN


def _score_to_hex(score: float, in_sun: bool) -> str:
    """Map sun score to hex colour for GeoJSON/Folium."""
    if not in_sun:
        return "#4a7fa5"   # muted blue — shade
    # Interpolate yellow (#FFD246) → orange (#FF8C00) based on score
    r = int(255)
    g = int(210 - (1 - score) * 80)
    b = int(70 - score * 70)
    return f"#{r:02x}{max(0,g):02x}{max(0,b):02x}"


# ── Terminal output ────────────────────────────────────────────────────────────

def print_header():
    print(f"\n{Fore.YELLOW}{'═'*54}")
    print(f"  ☀  SunCycle — Sun-aware cycling router")
    print(f"{'═'*54}{Style.RESET_ALL}")


def print_sun_bar(label: str, sun_percent: float, width: int = 24):
    """Render a coloured terminal progress bar for sun score."""
    filled = int(sun_percent / 100 * width)
    empty = width - filled

    if sun_percent >= 70:
        bar_col = Fore.YELLOW
        icon = "☀"
    elif sun_percent >= 45:
        bar_col = Fore.WHITE
        icon = "⛅"
    else:
        bar_col = Fore.BLUE
        icon = "☁"

    bar = bar_col + "█" * filled + Style.DIM + "░" * empty + Style.RESET_ALL
    pct = f"{bar_col}{sun_percent:.0f}%{Style.RESET_ALL}"
    print(f"  {icon} {label:<12} [{bar}] {pct}")


def print_route_summary(
    score: RouteScore,
    route: RouteCandidate,
    tag: str = "",
):
    """Print a compact route summary."""
    tag_str = f" {Fore.YELLOW}← {tag}{Style.RESET_ALL}" if tag else ""
    print(f"\n  {Fore.WHITE}{score.route_id.upper()}{Style.RESET_ALL}{tag_str}")
    print(f"  Distance:   {route.distance_km:.1f} km  |  "
          f"Est. time: {route.duration_min:.0f} min")
    print(f"  Sunny:      {score.sun_percent:.0f}%  |  "
          f"Sunny distance: {score.sunny_distance_m/1000:.1f} km")
    print(f"  Continuity: {score.continuity_score*100:.0f}%  |  "
          f"Intensity: {score.intensity_score*100:.0f}%")
    print(f"  Score:      {score.final_score*100:.1f}/100  — {score.label}")
    print_sun_bar(score.route_id, score.sun_percent)


def print_segment_map(score: RouteScore, cols: int = 60):
    """
    Render a single-line ASCII minimap of sun/shade segments.

    Each character represents an equal fraction of the route.
    █ yellow = sunny, ░ blue = shaded.
    Gives an instant visual of where the sunny stretches are.
    """
    n = len(score.point_scores)
    chars = []
    for pt in score.point_scores:
        col = _sun_colour_terminal(pt.sun_score, pt.in_sun)
        char = "█" if pt.in_sun else "░"
        chars.append(col + char)

    # Downsample to `cols` characters
    step = max(1, n // cols)
    sampled = chars[::step][:cols]

    print(f"\n  Route sun map (yellow=sun, blue=shade):")
    print(f"  A {''.join(sampled)}{Style.RESET_ALL} B")


def print_ranked_summary(ranked: RankedRoutes, routes: list[RouteCandidate]):
    """Print the full ranked output for all routes."""
    route_map = {r.id: r for r in routes}

    print(f"\n{Fore.YELLOW}{'─'*54}")
    print(f"  ROUTE RECOMMENDATIONS")
    print(f"{'─'*54}{Style.RESET_ALL}")

    seen = set()
    for score, tag in [
        (ranked.sunniest, "SUNNIEST"),
        (ranked.fastest, "FASTEST"),
        (ranked.balanced, "BALANCED"),
    ]:
        if score.route_id not in seen:
            route = route_map.get(score.route_id)
            if route:
                print_route_summary(score, route, tag=tag)
                print_segment_map(score)
            seen.add(score.route_id)

    # All routes
    print(f"\n{Fore.WHITE}  All candidates ranked:{Style.RESET_ALL}")
    for i, sc in enumerate(ranked.all_scores, 1):
        route = route_map.get(sc.route_id)
        dist = f"{route.distance_km:.1f}km" if route else "?"
        print(f"  {i}. {sc.route_id}  {sc.sun_percent:.0f}% sun  "
              f"{dist}  score={sc.final_score*100:.0f}/100")


# ── GeoJSON export ─────────────────────────────────────────────────────────────

def export_geojson(
    scores: list[RouteScore],
    routes: list[RouteCandidate],
    output_path: str = "suncycle_routes.geojson",
) -> str:
    """
    Export all scored routes as a GeoJSON FeatureCollection.

    Each segment is a separate LineString feature with properties:
      - route_id, in_sun, sun_score, avg_score, colour

    Load this in QGIS, kepler.gl, or Mapbox Studio to visualise.
    """
    route_map = {r.id: r for r in routes}
    features = []

    for score in scores:
        route = route_map.get(score.route_id)
        if not route:
            continue

        # One feature per segment (run of same sun state)
        for seg in score.segments:
            seg_points = score.point_scores[seg.start_idx:seg.end_idx + 1]
            if len(seg_points) < 2:
                continue

            # GeoJSON uses [lng, lat] order
            coords = [[pt.lng, pt.lat] for pt in seg_points]
            colour = _score_to_hex(seg.avg_score, seg.in_sun)

            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords,
                },
                "properties": {
                    "route_id": score.route_id,
                    "in_sun": seg.in_sun,
                    "sun_score": round(seg.avg_score, 3),
                    "sun_percent": score.sun_percent,
                    "final_score": round(score.final_score, 3),
                    "colour": colour,
                    "segment_length": seg.length,
                },
            })

    geojson = {"type": "FeatureCollection", "features": features}
    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2)

    return output_path


# ── Folium interactive map ─────────────────────────────────────────────────────

def export_folium_map(
    scores: list[RouteScore],
    routes: list[RouteCandidate],
    ranked: RankedRoutes,
    departure_time: datetime,
    output_path: str = "suncycle_map.html",
) -> str:
    """
    Generate an interactive HTML map with colour-coded sun/shade route segments.

    HOW IT WORKS:
    - Each segment (run of sun or shade points) is drawn as a polyline
    - Sunny segments: yellow/orange gradient based on intensity
    - Shaded segments: muted blue
    - Clicking a segment shows its sun score and route ID in a popup
    - The sunniest route is highlighted; others are shown dimly

    The output is a self-contained HTML file — no server needed.
    Open it in any browser.
    """
    try:
        import folium
    except ImportError:
        print("  ⚠ folium not installed. Run: pip3 install folium")
        return ""

    route_map = {r.id: r for r in routes}

    # Centre map on route midpoint
    all_lats = [pt.lat for sc in scores for pt in sc.point_scores]
    all_lngs = [pt.lng for sc in scores for pt in sc.point_scores]
    centre = (sum(all_lats)/len(all_lats), sum(all_lngs)/len(all_lngs))

    m = folium.Map(
        location=centre,
        zoom_start=14,
        tiles="CartoDB dark_matter",
    )

    sunniest_id = ranked.sunniest.route_id

    for score in scores:
        route = route_map.get(score.route_id)
        if not route:
            continue

        is_featured = (score.route_id == sunniest_id)
        opacity = 0.9 if is_featured else 0.35
        weight = 5 if is_featured else 3

        for seg in score.segments:
            seg_points = score.point_scores[seg.start_idx:seg.end_idx + 1]
            if len(seg_points) < 2:
                continue

            coords = [(pt.lat, pt.lng) for pt in seg_points]
            colour = _score_to_hex(seg.avg_score, seg.in_sun)
            state = "☀ Sunny" if seg.in_sun else "☁ Shaded"

            popup_html = f"""
            <div style='font-family:sans-serif; font-size:13px; padding:6px;'>
              <b>{score.route_id}</b> — {state}<br>
              Sun score: <b>{seg.avg_score*100:.0f}%</b><br>
              Route sun: <b>{score.sun_percent:.0f}%</b><br>
              Final score: <b>{score.final_score*100:.0f}/100</b>
            </div>
            """

            folium.PolyLine(
                locations=coords,
                color=colour,
                weight=weight,
                opacity=opacity,
                popup=folium.Popup(popup_html, max_width=200),
                tooltip=f"{score.route_id}: {state} ({seg.avg_score*100:.0f}%)",
            ).add_to(m)

    # Title box
    title_html = f"""
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                background: rgba(26,26,46,0.9); color: #FFD246; padding: 10px 20px;
                border-radius: 20px; font-family: sans-serif; font-size: 14px;
                font-weight: 500; z-index: 1000; border: 1px solid rgba(255,210,70,0.3);">
      ☀ SunCycle — {departure_time.strftime('%H:%M, %d %b %Y')}
      &nbsp;|&nbsp; Sunniest: {ranked.sunniest.sun_percent:.0f}%
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    m.save(output_path)
    return output_path
