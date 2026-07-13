"""
scoring.py — Multi-factor sun-aware route scoring
==================================================

SCORING FORMULA (weighted composite):

  final_score = (
      0.40 * sun_fraction        # % of points in direct sun
    + 0.25 * intensity_score     # avg solar intensity when in sun (sin altitude)
    + 0.20 * continuity_score    # longest sunny run / total points
    + 0.15 * (1 - detour_frac)   # distance penalty vs shortest route
  )

WHY CONTINUITY?
  Two routes both 50% sunny score the same on sun_fraction alone.
  But one might have a single 300m sunny stretch (great) vs 60 alternating
  5m patches (miserable). Continuity = longest_sunny_run / total_points
  captures this. A perfect sunny route scores 1.0; a flickering one → ~0.01.

WHY DETOUR PENALTY?
  A 3km sunniest route beats a 15km sunniest route between the same points.
  We normalise detour against the shortest candidate (not absolute distance)
  so the penalty is relative to what's achievable, not arbitrary.
"""

from dataclasses import dataclass, field
from datetime import datetime

from solar import SunCache, SunPosition
from shadows import ShadowEngine, fetch_buildings
from routing import Route


@dataclass
class PointResult:
    lat: float
    lng: float
    in_sun: bool
    score: float          # 0→1
    sun: SunPosition


@dataclass
class RouteScore:
    route_id: str
    point_results: list[PointResult]
    sun_fraction: float       # proportion of points in direct sun
    intensity_score: float    # avg intensity of sunny points
    continuity_score: float   # longest sunny run / total points
    detour_fraction: float    # extra distance vs shortest (0 = shortest)
    final_score: float        # weighted composite 0→1
    distance_m: float
    duration_s: float

    @property
    def sun_percent(self) -> float:
        return round(self.sun_fraction * 100, 1)

    @property
    def distance_km(self) -> float:
        return self.distance_m / 1000

    @property
    def duration_min(self) -> float:
        return self.duration_s / 60

    @property
    def grade(self) -> str:
        p = self.sun_percent
        if p >= 75: return "☀  Excellent"
        if p >= 55: return "⛅ Good"
        if p >= 35: return "🌤 Moderate"
        return "☁  Mostly shaded"

    @property
    def sunny_km(self) -> float:
        return round(self.distance_km * self.sun_fraction, 2)


def score_routes(
    routes: list[Route],
    departure: datetime,
    verbose: bool = True,
) -> list[RouteScore]:
    """
    Score all route candidates with real shadow-aware sun scoring.

    For each route:
      1. Fetch OSM buildings in the route corridor (Overpass API)
      2. Build a ShadowEngine with projected shadow polygons
      3. Test each sampled point: in shadow or not?
      4. Compute composite score
    """
    if not routes:
        return []

    sun_cache = SunCache()
    shortest_m = min(r.distance_m for r in routes)
    results = []

    for route in routes:
        if verbose:
            print(f"\n  Scoring {route.id} ({route.distance_km:.1f} km)...")

        # ── Get sun position for route centre ──────────────────────────────
        mid = route.sampled_points[len(route.sampled_points) // 2]
        sun = sun_cache.get(mid[0], mid[1], departure)

        if verbose:
            status = f"altitude={sun.altitude_deg:.1f}° az={sun.azimuth_deg:.0f}° ({sun.compass()})"
            print(f"    ☀  Sun: {status}")

        # ── Fetch buildings for this route ──────────────────────────────────
        bbox = route.bounding_box(buffer_m=150)
        if verbose:
            print(f"    🏢 Fetching OSM buildings...")
        buildings = fetch_buildings(*bbox)
        if verbose:
            print(f"    🏢 {len(buildings)} buildings loaded")

        # ── Build shadow engine ─────────────────────────────────────────────
        engine = ShadowEngine(buildings, sun)

        # ── Score each sampled point ────────────────────────────────────────
        point_results: list[PointResult] = []
        for i, (lat, lng) in enumerate(route.sampled_points):
            pt_sun = sun_cache.get(lat, lng, departure)
            score = engine.score_point(lat, lng)
            in_sun = score > 0.05 and pt_sun.is_daylight
            point_results.append(PointResult(
                lat=lat, lng=lng,
                in_sun=in_sun,
                score=score,
                sun=pt_sun,
            ))
            if verbose and i % 40 == 0:
                icon = "☀" if in_sun else "▓"
                print(f"    [{icon}] {i+1}/{len(route.sampled_points)}", end="\r")

        if verbose:
            print()

        # ── Aggregate ───────────────────────────────────────────────────────
        n = len(point_results)
        sunny = [p for p in point_results if p.in_sun]
        sun_fraction = len(sunny) / n if n else 0.0
        intensity = (sum(p.score for p in sunny) / len(sunny)) if sunny else 0.0
        continuity = _continuity(point_results, n)
        detour = min((route.distance_m - shortest_m) / max(shortest_m, 1), 1.0)

        final = (
            0.40 * sun_fraction
            + 0.25 * intensity
            + 0.20 * continuity
            + 0.15 * (1 - detour)
        )

        results.append(RouteScore(
            route_id=route.id,
            point_results=point_results,
            sun_fraction=sun_fraction,
            intensity_score=intensity,
            continuity_score=continuity,
            detour_fraction=detour,
            final_score=final,
            distance_m=route.distance_m,
            duration_s=route.duration_s,
        ))

    return results


def _continuity(points: list[PointResult], total: int) -> float:
    """
    Longest unbroken run of sunny points / total points.
    Rewards routes with long sunny stretches over flickering ones.
    """
    if total == 0:
        return 0.0
    best = cur = 0
    for p in points:
        if p.in_sun:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best / total


@dataclass
class Ranking:
    sunniest: RouteScore
    fastest: RouteScore
    balanced: RouteScore
    all_scores: list[RouteScore]


def rank(scores: list[RouteScore], routes: list[Route]) -> Ranking:
    """
    Identify three recommended routes:
      SUNNIEST  — highest composite sun score
      FASTEST   — least detour from shortest GraphHopper route
      BALANCED  — harmonic mean of sun_fraction and (1-detour): rewards
                  routes that are decent on both axes without extremes
    """
    sunniest = max(scores, key=lambda s: s.final_score)
    fastest  = min(scores, key=lambda s: s.detour_fraction)

    def _balanced(s: RouteScore) -> float:
        a, b = s.sun_fraction, 1 - s.detour_fraction
        return 2*a*b/(a+b) if (a+b) > 0 else 0

    balanced = max(scores, key=_balanced)

    return Ranking(
        sunniest=sunniest,
        fastest=fastest,
        balanced=balanced,
        all_scores=sorted(scores, key=lambda s: s.final_score, reverse=True),
    )
