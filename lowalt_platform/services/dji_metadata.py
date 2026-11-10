from __future__ import annotations

from datetime import datetime
import math
import re
from pathlib import Path

from PIL import Image

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
VIDEO_SUFFIXES = {'.mp4', '.mov', '.avi', '.mkv'}
SRT_SUFFIXES = {'.srt'}
_NUMBER = r'([+-]?\d+(?:\.\d+)?)'
_GPS_RE = re.compile(r'GPS\s*\(\s*' + _NUMBER + r'\s*[,，]\s*' + _NUMBER + r'(?:\s*[,，]\s*' + _NUMBER + r')?\s*\)', re.I)
_SRT_TIME_RE = re.compile(r'(\d{2,}):(\d{2}):(\d{2})[,.](\d{3})')
_SRT_TIMELINE_RE = re.compile(r'^\s*(\d{2,}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2,}:\d{2}:\d{2}[,.]\d{3})\s*$', re.M)


def valid_position(latitude: object, longitude: object) -> bool:
    try:
        return (latitude is not None and longitude is not None
                and math.isfinite(float(latitude)) and math.isfinite(float(longitude))
                and -90 <= float(latitude) <= 90 and -180 <= float(longitude) <= 180)
    except (ValueError, TypeError, OverflowError):
        return False


def _exif_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode('utf-8', errors='replace')
    return str(value).strip('\x00 ').strip() or None


def _finite_number(value: object) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None


def _dms_to_degrees(values: object) -> float | None:
    if not isinstance(values, (tuple, list)) or len(values) != 3:
        return None
    parts = [_finite_number(value) for value in values]
    if any(value is None for value in parts):
        return None
    degrees, minutes, seconds = parts
    if degrees < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        return None
    return degrees + minutes / 60 + seconds / 3600


def _iso_capture_time(value: object, offset: object = None, subsecond: object = None) -> str | None:
    text = _exif_text(value)
    if not text:
        return None
    try:
        result = datetime.strptime(text, '%Y:%m:%d %H:%M:%S').isoformat()
    except ValueError:
        return None
    fraction = _exif_text(subsecond)
    if fraction and fraction.isdigit():
        result += '.' + fraction
    zone = _exif_text(offset)
    if zone and re.fullmatch(r'[+-](?:0\d|1[0-3]):[0-5]\d|[+-]14:00', zone):
        result += zone
    return result


def extract_image_metadata(path: Path) -> dict:
    """Read EXIF without guessing missing hemispheres, times or physical scale."""
    metadata = dict.fromkeys(('latitude', 'longitude', 'altitude_m', 'altitude_reference',
                              'captured_at', 'camera_make', 'camera_model', 'width', 'height'))
    metadata['warnings'] = []
    with Image.open(path) as image:
        metadata['width'], metadata['height'] = image.size
        exif = image.getexif()
        metadata['camera_make'] = _exif_text(exif.get(0x010F))
        metadata['camera_model'] = _exif_text(exif.get(0x0110))
        try:
            photo = exif.get_ifd(0x8769)
            gps = exif.get_ifd(0x8825)
        except (ValueError, TypeError, OSError, SyntaxError):
            photo, gps = {}, {}
            metadata['warnings'].append('malformed_exif_ifd')
        metadata['captured_at'] = _iso_capture_time(
            photo.get(0x9003) or exif.get(0x9003) or exif.get(0x0132),
            photo.get(0x9011) or exif.get(0x9011),
            photo.get(0x9291) or exif.get(0x9291),
        )
        if gps:
            latitude, longitude = _dms_to_degrees(gps.get(2)), _dms_to_degrees(gps.get(4))
            lat_ref, lon_ref = _exif_text(gps.get(1)), _exif_text(gps.get(3))
            if lat_ref in {'N', 'S'} and lon_ref in {'E', 'W'}:
                if latitude is not None and lat_ref == 'S':
                    latitude = -latitude
                if longitude is not None and lon_ref == 'W':
                    longitude = -longitude
                if valid_position(latitude, longitude):
                    metadata['latitude'], metadata['longitude'] = round(latitude, 7), round(longitude, 7)
            if metadata['latitude'] is None:
                metadata['warnings'].append('invalid_or_incomplete_exif_gps')
            altitude = _finite_number(gps.get(6))
            ref = gps.get(5)
            if isinstance(ref, bytes) and len(ref) == 1:
                ref = ref[0]
            if altitude is not None and altitude >= 0 and ref in (0, 1):
                metadata['altitude_m'] = round(altitude * (-1 if ref == 1 else 1), 1)
                metadata['altitude_reference'] = 'exif_sea_level'
    return metadata


def _tag_number(block: str, names: str) -> float | None:
    match = re.search(r'\b(?:' + names + r')\s*[:=]\s*' + _NUMBER + r'(?=\s|[\],;，)]|$)', block, re.I)
    return _finite_number(match.group(1)) if match else None


def parse_srt(text: str) -> list[dict]:
    """Supported forms: GPS(latitude, longitude, altitude), lat/lon and DJI
    [latitude: ...] [longitude: ...] [rel_alt: ... abs_alt: ...]. No axis guessing.
    Relative H/rel_alt is retained separately, never used as sea-level altitude.
    """
    records = []
    for block in re.split(r'\n\s*\n', text.replace('\r\n', '\n').lstrip('\ufeff')):
        timeline = _SRT_TIMELINE_RE.search(block)
        if not timeline:
            continue
        endpoints = [tuple(map(int, _SRT_TIME_RE.fullmatch(value).groups())) for value in timeline.groups()]
        if any(minutes >= 60 or seconds >= 60 for _, minutes, seconds, _ in endpoints):
            continue
        start, end = [hours * 3600 + minutes * 60 + seconds + millis / 1000
                      for hours, minutes, seconds, millis in endpoints]
        if end < start:
            continue
        latitude = _tag_number(block, 'latitude|lat|纬度')
        longitude = _tag_number(block, 'longitude|lon|lng|经度')
        altitude = _tag_number(block, 'abs_alt')
        altitude_reference = 'srt_absolute_unspecified_datum' if altitude is not None else None
        relative = _tag_number(block, 'rel_alt')
        gps_match = _GPS_RE.search(block)
        if gps_match:
            latitude, longitude = float(gps_match[1]), float(gps_match[2])
            if altitude is None and gps_match[3]:
                altitude = _finite_number(gps_match[3])
                altitude_reference = 'srt_unspecified'
        if altitude is None:
            alt_match = re.search(r'\bALT\s*' + _NUMBER + r'\s*m', block, re.I)
            if alt_match:
                altitude, altitude_reference = _finite_number(alt_match[1]), 'srt_unspecified'
        if relative is None:
            height = re.search(r'\bH\s*' + _NUMBER + r'\s*m', block, re.I)
            relative = _finite_number(height[1]) if height else None
        if not valid_position(latitude, longitude):
            latitude = longitude = None
        records.append({
            'time_seconds': start,
            'end_time_seconds': end,
            'latitude': latitude, 'longitude': longitude,
            'altitude_m': round(altitude, 1) if altitude is not None else None,
            'altitude_reference': altitude_reference,
            'relative_altitude_m': round(relative, 1) if relative is not None else None,
            'record_index': len(records),
        })
    return records


def srt_nearest_position(records: list[dict], time_seconds: float, tolerance: float = 5.0) -> tuple[dict | None, str]:
    """Nearest observation within tolerance; never claim interpolation occurred.

    srt_frame is a near observation (<=1 s), not guaranteed frame-exact GPS.
    All other accepted observations are explicitly srt_nearest.
    """
    if not math.isfinite(time_seconds) or time_seconds < 0 or not math.isfinite(tolerance) or tolerance < 0:
        return None, 'none'
    positioned = [record for record in records
                  if _finite_number(record.get('time_seconds')) is not None
                  and record['time_seconds'] >= 0
                  and valid_position(record.get('latitude'), record.get('longitude'))]
    if not positioned:
        return None, 'none'
    nearest = min(positioned, key=lambda record: abs(record['time_seconds'] - time_seconds))
    delta = abs(nearest['time_seconds'] - time_seconds)
    if delta > tolerance:
        return None, 'none'
    return nearest, 'srt_frame' if delta <= 1.0 else 'srt_nearest'
