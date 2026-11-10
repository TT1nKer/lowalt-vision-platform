from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def make_fixture(source: Path) -> dict:
    """Create two distant flights and an unpaired clip to expose SRT cross-talk.

    These pixels and all coordinates are authored synthetic fixtures. They do
    not describe a real flight, parking facility, annotation or inference run.
    """
    source.mkdir(parents=True, exist_ok=True)
    image = Image.new('RGB', (160, 96), (32, 48, 64))
    ImageDraw.Draw(image).text((8, 40), 'SYNTHETIC FIXTURE', fill='white')
    exif = Image.Exif()
    exif[0x010F], exif[0x0110] = 'SYNTHETIC', 'EXIF fixture'
    exif[0x8769] = {0x9003: '2026:01:02 03:04:05', 0x9011: '-05:00', 0x9291: '125'}
    exif[0x8825] = {1: 'S', 2: (33., 30., 0.), 3: 'W', 4: (70., 30., 0.), 5: b'\x01', 6: 12.5}
    image.save(source / 'positioned.jpg', exif=exif)
    image.save(source / 'unpositioned.jpg')
    for name, channel in [('flight_a', 0), ('flight_b', 1), ('unpaired', 2)]:
        writer = cv2.VideoWriter(str(source / f'{name}.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 10., (160, 96))
        if not writer.isOpened():
            raise RuntimeError('MJPG VideoWriter unavailable')
        try:
            for index in range(30):
                frame = np.zeros((96, 160, 3), dtype=np.uint8)
                frame[:, :, channel] = 80 + index
                cv2.putText(frame, 'SYNTHETIC', (6, 45), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
                writer.write(frame)
        finally:
            writer.release()
    modern, legacy = [], []
    for second in range(3):
        timeline = f'{second + 1}\n00:00:0{second},000 --> 00:00:0{second},999\n'
        modern.append(timeline + f'[latitude: {43 + second / 1000:.3f}] [longitude: {-79 - second / 1000:.3f}] [rel_alt: 20 abs_alt: 120]\n')
        legacy.append(timeline + f'GPS ({-33 - second / 1000:.3f}, {151 + second / 1000:.3f}, 45), H 10m\n')
    (source / 'flight_a.SRT').write_text('\n'.join(modern), encoding='utf-8')
    (source / 'flight_b.srt').write_text('\n'.join(legacy), encoding='utf-8')
    return {
        'positioned.jpg': [-70.5, -33.5],
        'flight_a.avi': [[-79 - second / 1000, 43 + second / 1000] for second in range(3)],
        'flight_b.avi': [[151 + second / 1000, -33 - second / 1000] for second in range(3)],
    }
