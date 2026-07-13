"""
routing.py — Route fetching + OSM building geometry
====================================================

WHY MULTIPLE ROUTES?
The original code fetched one route and scored it. But "sunniest route" is
meaningless without alternatives to compare against. A route 10% longer but
80% sunnier is clearly better on a nice day. We need options.

HOW GRAPHHOPPER ALTERNATIVES WORK:
GraphHopper's free API supports `algorithm=alternative_route` which returns
up to 3 geometrically diverse paths. They're guaranteed to be meaningfully
different (not just minor variations) via a plateau/sharing ratio parameter.

WHY OSM BUILDINGS VIA OVERPASS?
ShadeMap's JS library runs in a browser via WebGL — it can't be called from
Python. For a Python backend, the next best option is:
  1. Query Overpass API for building footprints + heights along the route
  2. Run our own shadow projection (solar.py)

This gives us real building shadow geometry, free, offline-capable.
The tradeoff: Overpass can be slow (~2-5s for a bounding box query).
We mitigate by querying once per route bounding box (not per point).

BUILDING HEIGHT ESTIMATION:
OSM has three relevant tags:
  - `height`: explicit height in metres (most accurate, ~30% of UK buildings)
  - `building:levels`: floor count (we multiply by 3.2m average floor height)
  - neither: we fall back to a typology estimate based on `building` tag value
    (e.g. "house"→6m, "apartments"→15m, "office"→20m, "retail"→5m)
"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import requests
from shapely.geometry import LineString, Point


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class RouteCandidate:
    """A single candidate cycling route."""
    id: str                              # e.g. "route_0", "route_1"
    coordinates: list[tuple[float, float]]  # (lat, lng) pairs
    distance_m: float
    duration_s: float
    sampled_points: list[tuple[float, float]] = field(default_factory=list)
    buildings: list[dict] = field(default_factory=list)

    @property
    def distance_km(self) -> float:
        return self.distance_m / 1000

    @property
    def duration_min(self) -> float:
        return self.duration_s / 60

    def bounding_box(self, buffer_m: float = 150) -> dict:
        """
        Axis-aligned bounding box around the route with a buffer.
        Buffer = 150m because max shadow length we consider is 200m,
        so buildings just outside the route corridor can still cast shadows onto it.
        """
        lats = [pt[0] for pt in self.coordinates]
        lngs = [pt[1] for pt in self.coordinates]

        buf_deg_lat = buffer_m / 111320
        buf_deg_lng = buffer_m / (111320 * math.cos(math.radians(sum(lats)/len(lats))))

        return {
            "south": min(lats) - buf_deg_lat,
            "north": max(lats) + buf_deg_lat,
            "west":  min(lngs) - buf_deg_lng,
            "east":  max(lngs) + buf_deg_lng,
        }


@dataclass
class Building:
    """An OSM building with footprint and estimated height."""
    osm_id: str
    geometry: list[tuple[float, float]]  # ring of (lng, lat) coords
    height_m: float
    building_type: str


# ── Postcode geocoding ─────────────────────────────────────────────────────────

def geocode_postcode(postcode: str) -> tuple[float, float]:
    """
    Convert a UK postcode to (lat, lng) via postcodes.io.
    Free, no API key, covers all GB postcodes.
    """
    url = f"https://api.postcodes.io/postcodes/{postcode.replace(' ', '').upper()}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    result = r.json()["result"]
    return result["latitude"], result["longitude"]


# ── GraphHopper routing ────────────────────────────────────────────────────────

GRAPHHOPPER_URL = "https://graphhopper.com/api/1/route"

def fetch_route_candidates(
    start: tuple[float, float],
    end: tuple[float, float],
    api_key: str,
    max_alternatives: int = 3,
) -> list[RouteCandidate]:
    """
    Fetch multiple alternative cycling routes from GraphHopper.

    WHY `alternative_route` ALGORITHM?
    GraphHopper's standard Dijkstra returns one optimal path.
    `alternative_route` uses a penalty-based approach to find paths that
    share < 50% of edges with the primary route — ensuring genuine diversity.

    TRADEOFF: Alternative routes may be 10-40% longer than optimal.
    That's fine — we're optimising for sun, not pure speed.
    """
    params = {
        "point": [
            f"{start[0]},{start[1]}",
            f"{end[0]},{end[1]}"
        ],
        "profile": "bike",
        "algorithm": "alternative_route",
        "alternative_route.max_paths": max_alternatives,
        "alternative_route.max_weight_factor": 1.6,   # allow up to 60% longer
        "alternative_route.max_share_factor": 0.6,    # must differ by 40%+
        "points_encoded": False,
        "key": api_key,
    }

    r = requests.get(GRAPHHOPPER_URL, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()

    candidates = []
    for i, path in enumerate(data["paths"]):
        raw_coords = path["points"]["coordinates"]
        # GraphHopper returns [lng, lat] — flip to (lat, lng)
        coords = [(lat, lng) for lng, lat in raw_coords]
        candidates.append(RouteCandidate(
            id=f"route_{i}",
            coordinates=coords,
            distance_m=path["distance"],
            duration_s=path["time"] / 1000,  # GraphHopper returns milliseconds
        ))

    return candidates


# ── Route point sampling ───────────────────────────────────────────────────────

def sample_route(
    route: RouteCandidate,
    num_points: int = 150,
) -> list[tuple[float, float]]:
    """
    Sample evenly-spaced points along a route polyline using Shapely.

    WHY 150 POINTS?
    At 150 points per route and 3 routes, we're scoring 450 points total.
    Each shadow test against ~20 nearby buildings = ~9,000 geometric operations.
    At ~50µs per Shapely operation that's ~450ms — acceptable.
    Fewer points miss short shaded/sunny stretches; more points get slow.

    WHY SHAPELY interpolate()?
    Geographic routes are not evenly spaced — GraphHopper returns more vertices
    around curves and intersections. interpolate() resamples by arc length,
    giving us geometrically uniform coverage regardless of vertex density.
    """
    from shapely.geometry import LineString

    # Shapely uses (x, y) = (lng, lat)
    line = LineString([(lng, lat) for lat, lng in route.coordinates])
    total_length = line.length

    points = []
    for i in range(num_points):
        fraction = i / (num_points - 1)
        pt = line.interpolate(fraction * total_length)
        points.append((pt.y, pt.x))  # back to (lat, lng)

    return points


# ── OSM building fetching ──────────────────────────────────────────────────────

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Building type → estimated height fallback (metres)
BUILDING_HEIGHT_FALLBACK = {
    "house": 7,
    "detached": 7,
    "semi": 7,
    "terrace": 9,
    "apartments": 18,
    "residential": 12,
    "office": 20,
    "commercial": 8,
    "retail": 5,
    "supermarket": 6,
    "warehouse": 8,
    "industrial": 8,
    "church": 15,
    "school": 10,
    "hospital": 20,
    "hotel": 25,
    "yes": 10,         # generic building tag
    "": 10,
}


def estimate_building_height(tags: dict) -> float:
    """
    Estimate building height from OSM tags.

    Priority:
      1. Explicit `height` tag (most accurate)
      2. `building:levels` × 3.2m (typical floor-to-floor height)
      3. Type-based fallback from BUILDING_HEIGHT_FALLBACK
      4. 10m default
    """
    if "height" in tags:
        try:
            return float(tags["height"].replace("m", "").strip())
        except ValueError:
            pass

    if "building:levels" in tags:
        try:
            return float(tags["building:levels"]) * 3.2
        except ValueError:
            pass

    building_type = tags.get("building", "").lower()
    return BUILDING_HEIGHT_FALLBACK.get(building_type, 10.0)


def fetch_buildings_for_route(
    route: RouteCandidate,
    buffer_m: float = 150,
    max_retries: int = 2,
) -> list[dict]:
    """
    Query Overpass API for all buildings within buffer_m of the route.

    OVERPASS QUERY STRATEGY:
    We use a bounding box query (faster than `around:` polygon) because:
      - Bounding box queries hit Overpass cache more often
      - Our buffer is large enough that bbox vs polygon makes little difference
      - We filter out irrelevant buildings during scoring anyway

    RATE LIMITING:
    Overpass has a 1 req/sec soft limit. We add a small sleep between routes.
    For production, consider self-hosting Overpass or using the Overpass Turbo
    public instance with exponential backoff.

    RETURNS list of dicts with:
        id: OSM way ID
        geometry: list of (lng, lat) tuples (exterior ring)
        height: float metres
        building_type: str
    """
    bbox = route.bounding_box(buffer_m)
    query = f"""
[out:json][timeout:25];
way["building"]
  ({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
out body geom;
"""
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(OVERPASS_URL, data=query, timeout=30)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            if attempt == max_retries:
                print(f"    ⚠ Overpass query failed after {max_retries+1} attempts: {e}")
                return []
            time.sleep(2 ** attempt)

    buildings = []
    for element in data.get("elements", []):
        if element.get("type") != "way":
            continue
        geom = element.get("geometry", [])
        if len(geom) < 3:
            continue

        tags = element.get("tags", {})
        height = estimate_building_height(tags)
        coords = [(node["lon"], node["lat"]) for node in geom]

        buildings.append({
            "id": str(element["id"]),
            "geometry": coords,
            "height": height,
            "building_type": tags.get("building", "yes"),
        })

    return buildings

geocode = geocode_postcode
fetch_routes = fetch_route_candidates
sample_points = sample_route
Route = RouteCandidate
