import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from uav_data.ingest import download_registered_files
from uav_data.source_files import RegisteredFile, validate_download_budget, verify_file


def md5_checksum(content: bytes) -> str:
    return "md5:" + hashlib.md5(content).hexdigest()


class UavSourceIngestTests(unittest.TestCase):
    def test_https_transport_preserves_download_and_checksum_gates(self):
        content = b'{"source":"https fixture"}'

        class FakeHttpsResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size: int):
                self.chunk_size = chunk_size
                yield content[:8]
                yield content[8:]

        registered = RegisteredFile(
            name="scene.json",
            url="https://example.invalid/scene.json",
            size_bytes=len(content),
            checksum=md5_checksum(content),
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "uav_data.ingest.requests.get", return_value=FakeHttpsResponse()
        ):
            destination = Path(directory)

            downloaded = download_registered_files(
                (registered,), destination, maximum_bytes=len(content)
            )

            self.assertEqual(downloaded[0].read_bytes(), content)

    def test_cancellation_removes_partial_file(self):
        content = b"partial content"

        class InterruptedResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size: int):
                yield content
                raise KeyboardInterrupt()

        registered = RegisteredFile(
            name="scene.json",
            url="https://example.invalid/scene.json",
            size_bytes=len(content) * 2,
            checksum=md5_checksum(content * 2),
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "uav_data.ingest.requests.get", return_value=InterruptedResponse()
        ):
            destination = Path(directory)

            with self.assertRaises(KeyboardInterrupt):
                download_registered_files(
                    (registered,), destination, maximum_bytes=len(content) * 2
                )

            self.assertFalse((destination / "scene.json.part").exists())

    def test_budget_rejects_total_before_download(self):
        registered = RegisteredFile(
            name="sample.json",
            url="https://example.invalid/sample.json",
            size_bytes=11,
            checksum="md5:00000000000000000000000000000000",
        )

        with self.assertRaisesRegex(ValueError, "download budget"):
            validate_download_budget((registered,), maximum_bytes=10)

    def test_registered_name_cannot_escape_destination(self):
        with self.assertRaisesRegex(ValueError, "basename"):
            RegisteredFile(
                name="../sample.json",
                url="https://example.invalid/sample.json",
                size_bytes=1,
                checksum="md5:00000000000000000000000000000000",
            )

    def test_verify_file_rejects_wrong_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "sample.json")
            path.write_bytes(b"real content")
            registered = RegisteredFile(
                name=path.name,
                url=path.as_uri(),
                size_bytes=path.stat().st_size,
                checksum=md5_checksum(b"different content"),
            )

            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_file(path, registered)

    def test_valid_local_source_is_atomically_downloaded(self):
        content = b'{"scene":"DJI_0001"}'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "upstream.json"
            destination = root / "downloaded"
            source.write_bytes(content)
            registered = RegisteredFile(
                name="scene.json",
                url=source.as_uri(),
                size_bytes=len(content),
                checksum=md5_checksum(content),
            )

            downloaded = download_registered_files(
                (registered,), destination, maximum_bytes=len(content)
            )

            self.assertEqual(downloaded, (destination / "scene.json",))
            self.assertEqual(downloaded[0].read_bytes(), content)
            self.assertFalse((destination / "scene.json.part").exists())

    def test_checksum_failure_leaves_no_partial_or_destination_file(self):
        content = b"untrusted"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "upstream.json"
            destination = root / "downloaded"
            source.write_bytes(content)
            registered = RegisteredFile(
                name="scene.json",
                url=source.as_uri(),
                size_bytes=len(content),
                checksum=md5_checksum(b"trusted"),
            )

            with self.assertRaisesRegex(ValueError, "checksum"):
                download_registered_files(
                    (registered,), destination, maximum_bytes=len(content)
                )

            self.assertFalse((destination / "scene.json").exists())
            self.assertFalse((destination / "scene.json.part").exists())

    def test_existing_mismatched_file_is_not_overwritten(self):
        expected = b"expected"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "upstream.json"
            destination = root / "downloaded"
            destination.mkdir()
            source.write_bytes(expected)
            existing = destination / "scene.json"
            existing.write_bytes(b"different")
            registered = RegisteredFile(
                name="scene.json",
                url=source.as_uri(),
                size_bytes=len(expected),
                checksum=md5_checksum(expected),
            )

            with self.assertRaisesRegex(ValueError, "existing file"):
                download_registered_files(
                    (registered,), destination, maximum_bytes=len(expected)
                )

            self.assertEqual(existing.read_bytes(), b"different")

    def test_verified_existing_file_removes_stale_partial(self):
        content = b"verified"
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            target = destination / "scene.json"
            partial = destination / "scene.json.part"
            target.write_bytes(content)
            partial.write_bytes(b"stale")
            registered = RegisteredFile(
                name=target.name,
                url=target.as_uri(),
                size_bytes=len(content),
                checksum=md5_checksum(content),
            )

            downloaded = download_registered_files(
                (registered,), destination, maximum_bytes=len(content)
            )

            self.assertEqual(downloaded, (target,))
            self.assertFalse(partial.exists())


if __name__ == "__main__":
    unittest.main()
