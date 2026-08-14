import unittest

from uav_data.schema import (
    FrameRecord,
    GsdStatus,
    SequenceManifest,
    SequenceRecord,
    SourceKind,
    SpatialReference,
)


def make_sequence(
    sequence_id: str = "seq-a",
    source_kind: SourceKind = SourceKind.UAV_VIDEO_SEQUENCE,
) -> SequenceRecord:
    return SequenceRecord(
        source_id="research-source",
        site_id="site-a",
        sequence_id=sequence_id,
        source_kind=source_kind,
        license_id="research-only",
        spatial_reference=SpatialReference(
            crs="LOCAL:SITE_A_METRES",
            gsd_status=GsdStatus.UNKNOWN,
        ),
    )


def make_frame(
    frame_id: str = "frame-1",
    sequence_id: str = "seq-a",
    frame_index: int = 0,
    capture_time_s: float | None = 0.0,
) -> FrameRecord:
    return FrameRecord(
        frame_id=frame_id,
        sequence_id=sequence_id,
        frame_index=frame_index,
        media_path=f"frames/{frame_id}.png",
        width_px=3840,
        height_px=2160,
        capture_time_s=capture_time_s,
    )


class UavDataSchemaTests(unittest.TestCase):
    def test_uav_manifest_rejects_frame_from_another_sequence(self):
        with self.assertRaisesRegex(ValueError, "belongs to sequence"):
            SequenceManifest(
                sequence=make_sequence("seq-a"),
                frames=(make_frame(sequence_id="seq-b"),),
            )

    def test_static_orthophoto_may_explicitly_have_no_capture_time(self):
        manifest = SequenceManifest(
            sequence=make_sequence("wmts-1", SourceKind.STATIC_ORTHOPHOTO),
            frames=(),
        )

        self.assertEqual(manifest.sequence.source_kind, SourceKind.STATIC_ORTHOPHOTO)

    def test_duplicate_frame_index_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate frame_index"):
            SequenceManifest(
                sequence=make_sequence(),
                frames=(
                    make_frame("frame-1", frame_index=0),
                    make_frame("frame-2", frame_index=0),
                ),
            )

    def test_decreasing_capture_time_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "capture_time_s must not decrease"):
            SequenceManifest(
                sequence=make_sequence(),
                frames=(
                    make_frame("frame-1", frame_index=0, capture_time_s=2.0),
                    make_frame("frame-2", frame_index=1, capture_time_s=1.0),
                ),
            )

    def test_frame_rejects_nonfinite_gps(self):
        with self.assertRaisesRegex(ValueError, "latitude"):
            FrameRecord(
                frame_id="frame-1",
                sequence_id="seq-a",
                frame_index=0,
                media_path="frame.png",
                width_px=100,
                height_px=100,
                latitude=float("nan"),
                longitude=120.0,
            )

    def test_manifest_serializes_stable_split_key_and_enum_values(self):
        manifest = SequenceManifest(
            sequence=make_sequence(),
            frames=(make_frame(),),
        )

        serialized = manifest.to_dict()

        self.assertEqual(serialized["split_group"], "site-a/seq-a")
        self.assertEqual(serialized["sequence"]["source_kind"], "uav_video_sequence")
        self.assertEqual(serialized["sequence"]["spatial_reference"]["gsd_status"], "unknown")
        self.assertEqual(serialized["frames"][0]["frame_id"], "frame-1")


if __name__ == "__main__":
    unittest.main()
