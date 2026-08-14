from __future__ import annotations

import re
from pathlib import Path

from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}
SRT_SUFFIXES = {".srt"}

_GPS_RE = re.compile(
    r"GPS\s*\(\s*(-?\d+\.\d+)\s*[,，]\s*(-?\d+\.\d+)(?:\s*[,，]\s*(-?\d+(?:\.\d+)?))?\s*\)",
    re.IGNORECASE,
)
_ALT_RE = re.compile(r"(?:ALT|H)\s*(-?\d+(?:\.\d+)?)\s*m", re.IGNORECASE)
_SRT_TIME_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})")
_LAT_LON_RE = re.compile(
    r"(?:lat|纬度)\s*[:=]?\s*(-?\d+\.\d+)\s*[,，;]\s*(?:lon|lng|经度)\s*[:=]?\s*(-?\d+\.\d+)",
    re.IGNORECASE,
)


def _exif_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip("\x00 ").strip() or None
    text = str(value).strip("\x00 ").strip()
    return text or None


def _dms_to_degrees(values: object) -> float | None:
    if not isinstance(values, (tuple, list)) or len(values) != 3:
        return None
    try:
        degrees, minutes, seconds = (float(value) for value in values)
    except (TypeError, ValueError):
        return None
    return degrees + minutes / 60.0 + seconds / 3600.0


def _iso_capture_time(value: object) -> str | None:
    text = _exif_text(value)
    if not text:
        return None
    digits = re.sub(r"[^\d]", "", text)
    if len(digits) < 14:
        return None
    return f"{digits[0:4]}-{digits[4:6]}-{digits[6:8]}T{digits[8:10]}:{digits[10:12]}:{digits[12:14]}"


def extract_image_metadata(path: Path) -> dict:
    """Return the metadata an imported image can honestly provide.

    Every field is None when the media does not carry it; no values are
    invented. GPS comes from the EXIF GPS IFD (DJI writes it by default).
    """
    metadata: dict = {
        "latitude": None,
        "longitude": None,
        "altitude_m": None,
        "captured_at": None,
        "camera_make": None,
        "camera_model": None,
        "width": None,
        "height": None,
    }
    with Image.open(path) as image:
        metadata["width"], metadata["height"] = image.size
        exif = image.getexif()
        if exif:
            metadata["camera_make"] = _exif_text(exif.get(0x010F))
            metadata["camera_model"] = _exif_text(exif.get(0x0110))
            metadata["captured_at"] = _iso_capture_time(exif.get(0x9003) or exif.get(0x0132))
        gps = exif.get_ifd(0x8825) if hasattr(exif, "get_ifd") else {}
        if gps:
            latitude = _dms_to_degrees(gps.get(2))
            longitude = _dms_to_degrees(gps.get(4))
            if latitude is not None and str(gps.get(1)).strip().upper() == "S":
                latitude = -latitude
            if longitude is not None and str(gps.get(3)).strip().upper() == "W":
                longitude = -longitude
            altitude = gps.get(6)
            altitude_value = None
            if isinstance(altitude, (tuple, list)) and altitude:
                altitude_value = float(altitude[0])
            elif altitude is not None:
                try:
                    altitude_value = float(altitude)
                except (TypeError, ValueError):
                    altitude_value = None
            if altitude_value is not None:
                metadata["altitude_m"] = round(altitude_value, 1)
                if str(gps.get(5)).strip() == "1":
                    metadata["altitude_m"] = -metadata["altitude_m"]
            if latitude is not None:
                metadata["latitude"] = round(latitude, 7)
            if longitude is not None:
                metadata["longitude"] = round(longitude, 7)
    return metadata


def parse_srt(text: str) -> list[dict]:
    """Parse a DJI subtitle sidecar into time-tagged position records.

    DJI SRT blocks look like:

        1
        00:00:00,000 --> 00:00:00,500
        F/2.8, SS 1/50, ISO 100, EV 0, GPS (30.123456, 120.123456, 15.0), D 3.2m, H 2.0m

    Records that contain neither a timestamp nor coordinates are skipped;
    fields stay None when the sidecar does not carry them.
    """
    records: list[dict] = []
    blocks = re.split(r"\n\s*\n", text)
    for block in blocks:
        time_match = _SRT_TIME_RE.search(block)
        gps_match = _GPS_RE.search(block)
        lat_lon_match = _LAT_LON_RE.search(block)
        if not time_match and not gps_match and not lat_lon_match:
            continue
        time_seconds = None
        if time_match:
            hours, minutes, seconds, millis = (int(part) for part in time_match.groups())
            time_seconds = hours * 3600 + minutes * 60 + seconds + millis / 1000.0
        latitude = longitude = altitude_m = None
        if gps_match:
            latitude, longitude = float(gps_match.group(1)), float(gps_match.group(2))
            if gps_match.group(3):
                altitude_m = round(float(gps_match.group(3)), 1)
        elif lat_lon_match:
            latitude, longitude = float(lat_lon_match.group(1)), float(lat_lon_match.group(2))
        if altitude_m is None:
            altitude_match = _ALT_RE.search(block)
            if altitude_match:
                altitude_m = round(float(altitude_match.group(1)), 1)
        records.append(
            {
                "time_seconds": time_seconds,
                "latitude": latitude,
                "longitude": longitude,
                "altitude_m": altitude_m,
            }
        )
    return records


def srt_nearest_position(records: list[dict], time_seconds: float, tolerance: float = 5.0) -> tuple[dict | None, str]:
    """Return the nearest positioned SRT record and how it was matched."""
    positioned = [record for record in records if record.get("time_seconds") is not None and record.get("latitude") is not None]
    if not positioned:
        return None, "none"
    nearest = min(positioned, key=lambda record: abs(record["time_seconds"] - time_seconds))
    delta = abs(nearest["time_seconds"] - time_seconds)
    if delta <= 1.0:
        return nearest, "srt_frame"
    if delta <= tolerance:
        return nearest, "srt_interpolated"
    return None, "none"
