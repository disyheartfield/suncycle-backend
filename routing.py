"""Look up UK postcodes, fetch cycling routes and sample their geometry."""

import math
import re
from dataclasses import dataclass

import requests
from shapely.geometry import LineString


class RoutingError(Exception):
    """A safe message for the app, without request URLs or API keys."""

    def __init__(self, message, status_code=502):
        super().__init__(message)
        self.status_code = status_code


def _get_json(url, service, params=None, postcode_lookup=False):
    try:
        response = requests.get(url, params=params, timeout=(5, 20))
        if postcode_lookup and response.status_code in (400, 404):
            raise RoutingError("Postcode not found. Check both UK postcodes.", 400)
        response.raise_for_status()
        return response.json()
    except requests.Timeout:
        raise RoutingError(f"{service} timed out. Please try again.", 504) from None
    except requests.RequestException:
        raise RoutingError(f"Could not reach {service}. Check your connection and service access.") from None
    except ValueError:
        raise RoutingError(f"{service} returned an unreadable response.") from None


def _coordinate(lat, lng):
    lat, lng = float(lat), float(lng)
    if not (math.isfinite(lat) and math.isfinite(lng) and abs(lat) <= 90 and abs(lng) <= 180):
        raise ValueError("Invalid coordinate")
    return lat, lng


def geocode(postcode):
    """Return (latitude, longitude) for a UK postcode."""
    postcode = re.sub(r"\s+", "", postcode).upper()
    if not re.fullmatch(r"[A-Z0-9]{5,7}", postcode):
        raise RoutingError("Enter a valid UK postcode for both locations.", 400)
    data = _get_json(
        f"https://api.postcodes.io/postcodes/{postcode}",
        "Postcodes.io", postcode_lookup=True,
    )
    try:
        result = data["result"]
        return _coordinate(result["latitude"], result["longitude"])
    except (KeyError, TypeError, ValueError):
        raise RoutingError("Postcodes.io returned invalid location data.") from None


@dataclass
class Route:
    id: str
    coordinates: list[tuple[float, float]]  # (latitude, longitude)
    distance_m: float
    duration_s: float


def fetch_routes(start, end, api_key):
    """Ask GraphHopper for up to three cycling alternatives."""
    data = _get_json("https://graphhopper.com/api/1/route", "GraphHopper", params={
        "point": [f"{start[0]},{start[1]}", f"{end[0]},{end[1]}"],
        "profile": "bike",
        "algorithm": "alternative_route",
        "alternative_route.max_paths": 3,
        "alternative_route.max_weight_factor": 1.6,
        "alternative_route.max_share_factor": 0.6,
        "points_encoded": "false",
        "key": api_key,
    })
    try:
        routes = []
        for i, path in enumerate(data["paths"]):
            coords = [_coordinate(point[1], point[0]) for point in path["points"]["coordinates"]]
            distance, duration = float(path["distance"]), float(path["time"]) / 1000
            if len(coords) < 2 or not all(math.isfinite(v) and v > 0 for v in (distance, duration)):
                raise ValueError("Invalid route")
            routes.append(Route(f"route_{i}", coords, distance, duration))
        if not routes:
            raise RoutingError("No cycling route found between these postcodes.", 404)
        return routes
    except (KeyError, IndexError, TypeError, ValueError):
        raise RoutingError("GraphHopper returned invalid route data.") from None


def sample_points(route, count=120):
    """Preserve the app's sampling: 120 points spaced along the lon/lat line.

    Spacing is in coordinate degrees, not exact metres. Both endpoints remain.
    ShadeMap counts these samples; sending uneven raw vertices would bias it.
    """
    if count < 2:
        raise ValueError("At least two sample points are required")
    line = LineString([(lng, lat) for lat, lng in route.coordinates])
    if line.length == 0:
        raise RoutingError("The route has no usable length. Try different postcodes.", 400)
    points = [line.interpolate(i * line.length / (count - 1)) for i in range(count)]
    return [[point.x, point.y] for point in points]  # app expects [longitude, latitude]
