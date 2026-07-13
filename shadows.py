"""
shadows.py — OSM building shadows via 2.5D projection
======================================================

HOW SHADOW PROJECTION WORKS:
  1. Fetch building footprints from Overpass API for a bounding box
  2. For each building, compute shadow length = height / tan(sun altitude)
  3. Displace each footprint vertex in the shadow direction (opposite sun azimuth)
  4. The shadow polygon = convex hull of (original footprint + displaced vertices)
  5. Test each route point: does any shadow polygon contain it?

This is the same geometry ShadeMap uses, implemented in pure Python with Shapely.
"""

import math
import time
import requests
from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree
from typing import Optional
from solar import SunPosition


OVERPASS_MIRRORS = [
    "https://overpass-api.fr/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.de/api/interpreter",
]

# Building type → fallback height in metres
_HEIGHT_FALLBACK = {
    "house": 7, "detached": 7, "semidetached_house": 7, "terrace": 9,
    "apartments": 18, "residential": 12, "office": 20, "commercial": 8,
    "retail": 5, "supermarket": 7, "warehouse": 8, "industrial": 8,
    "church": 14, "school": 10, "hospital": 20, "hotel": 25,
    "university": 15, "civic": 12, "yes": 10,
}

# Degree scales at London latitude (~51.5°)
_LAT_SCALE = 1 / 111320
_LNG_SCALE = 1 / (111320 * math.cos(math.radians(51.5)))


def _building_height(tags: dict) -> float:
    for key in ("height", "building:height"):
        if key in tags:
            try:
                return float(str(tags[key]).replace("m","").strip())
            except ValueError:
                pass
    if "building:levels" in tags:
        try:
            return float(tags["building:levels"]) * 3.2
        except ValueError:
            pass
    return _HEIGHT_FALLBACK.get(tags.get("building","yes").lower(), 10.0)


def fetch_buildings(south, west, north, east):
    return []

def _shadow_polygon(coords: list[tuple], height: float, sun: SunPosition) -> Optional[Polygon]:
    """
    Project a building footprint into a shadow polygon on the ground.

    Shadow direction = sun azimuth + 180° (shadows fall away from sun).
    Shadow length    = height / tan(sun altitude)
    We displace each vertex by (shadow_length × sin/cos of shadow direction)
    then take the convex hull to avoid self-intersections.
    """
    if sun.altitude_deg < 2.0:
        return None

    sh_len = min(height / math.tan(sun.altitude_rad), 250.0)
    shadow_az = (sun.azimuth_deg + 180) % 360
    shadow_az_rad = math.radians(shadow_az)

    # Convert metres to degrees (approximate, valid for <500m)
    dx = sh_len * math.sin(shadow_az_rad) * _LNG_SCALE
    dy = sh_len * math.cos(shadow_az_rad) * _LAT_SCALE

    displaced = [(x + dx, y + dy) for x, y in coords]
    all_coords = coords + displaced

    try:
        hull = Polygon(all_coords).convex_hull
        return hull if hull.is_valid and not hull.is_empty else None
    except Exception:
        return None


class ShadowEngine:
    """
    Builds shadow polygons for a set of buildings at a given sun position,
    then efficiently tests points using an STRtree spatial index.

    Usage:
        engine = ShadowEngine(buildings, sun)
        in_shadow = engine.point_in_shadow(lat, lng)
    """

    def __init__(self, buildings: list[dict], sun: SunPosition):
        self.sun = sun
        self._shadows: list[Polygon] = []

        if not sun.is_daylight or sun.altitude_deg < 2.0:
            self._tree = None
            return

        for b in buildings:
            poly = _shadow_polygon(b["coords"], b["height"], sun)
            if poly and poly.is_valid and not poly.is_empty:
                self._shadows.append(poly)

        self._tree = STRtree(self._shadows) if self._shadows else None

    def point_in_shadow(self, lat: float, lng: float) -> bool:
        """Return True if this point falls inside any building shadow."""
        if self._tree is None:
            return not self.sun.is_daylight

        pt = Point(lng, lat)
        # STRtree.query returns candidate indices whose bounding boxes intersect
        candidates = self._tree.query(pt)
        for idx in candidates:
            if self._shadows[idx].contains(pt):
                return True
        return False

    def score_point(self, lat: float, lng: float) -> float:
        """
        Return sun score 0→1 for a point.
        In shadow  → 0.05 (diffuse sky light only)
        In sun     → sin(altitude) — full intensity at noon, low at dawn/dusk
        """
        if not self.sun.is_daylight:
            return 0.0
        if self.point_in_shadow(lat, lng):
            return 0.05  # diffuse sky radiation — not zero but heavily penalised
        return self.sun.intensity
