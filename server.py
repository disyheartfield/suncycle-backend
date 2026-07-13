"""
server.py — SunCycle FastAPI backend
=====================================
Wraps the existing Python engine as a clean REST API.
The mobile app talks only to this — no Python logic in the frontend.

Run:
    uvicorn server:app --reload --port 8000
"""

import sys
import os
from datetime import datetime
from typing import Optional

# Add suncycle engine to path — adjust if your files are elsewhere
SUNCYCLE_PATH = os.path.join(os.path.dirname(__file__), "..", "suncycle")
sys.path.insert(0, SUNCYCLE_PATH)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="SunCycle API", version="1.0.0")

# Allow requests from Expo dev server and any local origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GRAPHHOPPER_API_KEY = "ae661b79-172e-40b8-9366-9f02f6bb04ca"


# ── Request / Response models ──────────────────────────────────────────────────

class RouteRequest(BaseModel):
    start: str          # UK postcode e.g. "N19 3DA"
    end: str            # UK postcode e.g. "SE1 7PB"
    departure_time: Optional[str] = None   # "HH:MM" 24hr, None = now


class SegmentData(BaseModel):
    coordinates: list[list[float]]   # [[lng, lat], ...]
    in_sun: bool
    sun_score: float
    colour: str


class RouteResult(BaseModel):
    id: str
    distance_km: float
    duration_min: float
    sun_percent: float
    sunny_km: float
    score: float
    grade: str
    description: str
    segments: list[SegmentData]
    is_sunniest: bool
    is_fastest: bool


class RouteResponse(BaseModel):
    routes: list[RouteResult]
    sun_position: dict      # azimuth, altitude, compass
    departure_time: str
    start_coords: list[float]
    end_coords: list[float]


# ── Route description generator ────────────────────────────────────────────────

def _describe_route(sun_pct: float, altitude: float, duration_min: float) -> str:
    """
    Generate a short, warm, human description of the ride.
    This is the emotional layer — makes the app feel alive.
    """
    time_of_day = (
        "morning" if 5 <= datetime.now().hour < 12
        else "afternoon" if 12 <= datetime.now().hour < 17
        else "evening"
    )

    if sun_pct >= 80:
        openers = [
            f"Golden {time_of_day} ride",
            f"Warm light all the way",
            f"Sun on your face the whole way",
        ]
    elif sun_pct >= 60:
        openers = [
            f"Mostly sunny {time_of_day}",
            f"Good stretches of open sun",
            f"Bright route through the city",
        ]
    elif sun_pct >= 40:
        openers = [
            f"Mix of sun and shade",
            f"Some lovely sunny stretches",
            f"Partly open, partly sheltered",
        ]
    else:
        openers = [
            f"Shaded route — cooler ride",
            f"Mostly sheltered by buildings",
            f"Good for a hot day",
        ]

    import hashlib
    idx = int(hashlib.md5(f"{sun_pct}{duration_min}".encode()).hexdigest(), 16) % len(openers)
    return openers[idx]


def _hex_colour(score: float, in_sun: bool) -> str:
    if not in_sun:
        return "#4a7fa5"
    g = int(210 - (1 - score) * 80)
    b = int(max(0, 70 - score * 70))
    return f"#ff{max(0, g):02x}{b:02x}"


# ── Main endpoint ──────────────────────────────────────────────────────────────

@app.post("/routes", response_model=RouteResponse)
async def get_routes(req: RouteRequest):
    """
    Main route endpoint.
    Takes start/end postcodes + optional departure time.
    Returns scored, ranked routes with segment-level sun data.
    """
    try:
        from routing import geocode, fetch_routes, sample_points
        from scoring import score_routes, rank
        from solar import SunCache
    except ImportError as e:
        raise HTTPException(500, f"SunCycle engine not found: {e}. "
                                 f"Check SUNCYCLE_PATH in server.py points to your suncycle folder.")

    # ── Parse departure time ───────────────────────────────────────────────────
    if req.departure_time:
        h, m = map(int, req.departure_time.split(":"))
        departure = datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)
    else:
        departure = datetime.now()

    # ── Geocode ────────────────────────────────────────────────────────────────
    try:
        start_coords = geocode(req.start.upper())
        end_coords   = geocode(req.end.upper())
    except Exception as e:
        raise HTTPException(400, f"Postcode lookup failed: {e}")

    # ── Fetch routes ───────────────────────────────────────────────────────────
    try:
        routes = fetch_routes(start_coords, end_coords, GRAPHHOPPER_API_KEY, max_alternatives=3)
    except Exception as e:
        raise HTTPException(502, f"Routing failed: {e}")

    # ── Sample + score ─────────────────────────────────────────────────────────
    for route in routes:
        route.sampled_points = sample_points(route, 120)

    scores = score_routes(routes, departure, verbose=False)
    ranking = rank(scores, routes)

    # ── Get sun position ───────────────────────────────────────────────────────
    mid_lat = (start_coords[0] + end_coords[0]) / 2
    mid_lng = (start_coords[1] + end_coords[1]) / 2
    sun = SunCache().get(mid_lat, mid_lng, departure)

    # ── Build response ─────────────────────────────────────────────────────────
    route_map = {r.id: r for r in routes}
    result_routes = []

    for score in ranking.all_scores:
        route = route_map[score.route_id]

        # Build segments from point results
        segments = []
        pts = score.point_results
        if pts:
            run_start = 0
            for i in range(1, len(pts) + 1):
                end_of_run = (i == len(pts)) or (pts[i].in_sun != pts[run_start].in_sun)
                if end_of_run:
                    run = pts[run_start:i]
                    avg = sum(p.score for p in run) / len(run)
                    segments.append(SegmentData(
                        coordinates=[[p.lng, p.lat] for p in run],
                        in_sun=run[0].in_sun,
                        sun_score=round(avg, 3),
                        colour=_hex_colour(avg, run[0].in_sun),
                    ))
                    run_start = i

        result_routes.append(RouteResult(
            id=score.route_id,
            distance_km=round(score.distance_km, 1),
            duration_min=round(score.duration_min),
            sun_percent=score.sun_percent,
            sunny_km=score.sunny_km,
            score=round(score.final_score * 100, 1),
            grade=score.grade,
            description=_describe_route(score.sun_percent, sun.altitude_deg, score.duration_min),
            segments=segments,
            is_sunniest=(score.route_id == ranking.sunniest.route_id),
            is_fastest=(score.route_id == ranking.fastest.route_id),
        ))

    return RouteResponse(
        routes=result_routes,
        sun_position={
            "azimuth": round(sun.azimuth_deg, 1),
            "altitude": round(sun.altitude_deg, 1),
            "compass": sun.compass_label(),
            "intensity": round(sun.intensity, 2),
        },
        departure_time=departure.strftime("%H:%M"),
        start_coords=[start_coords[0], start_coords[1]],
        end_coords=[end_coords[0], end_coords[1]],
    )


@app.get("/health")
async def health():
    return {"status": "ok", "service": "SunCycle API"}
