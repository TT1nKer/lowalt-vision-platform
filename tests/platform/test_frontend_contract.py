from __future__ import annotations

from pathlib import Path
import unittest


WEB_ROOT = Path(__file__).parents[2] / "lowalt_platform" / "web"


class FrontendContractTests(unittest.TestCase):
    def test_map_uses_native_blocks_and_has_a_media_viewer(self) -> None:
        index = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
        script = (WEB_ROOT / "platform.js").read_text(encoding="utf-8")

        self.assertIn('id="mediaViewer"', index)
        self.assertIn("/api/platform/imagery/blocks?", script)
        self.assertIn("openMediaViewer", script)
        self.assertIn("fitBounds(layer.getBounds()", script)
        self.assertIn("/secondary", script)
        self.assertIn('id="secondaryEvidence"', index)

        engineering = (WEB_ROOT / "engineering.html").read_text(encoding="utf-8")
        self.assertIn('id="secondaryProgress"', engineering)
        self.assertIn("secondary_analysis.completed", script)

    def test_engineering_is_an_operational_console(self) -> None:
        engineering = (WEB_ROOT / "engineering.html").read_text(encoding="utf-8")

        self.assertNotIn("Easy continue", engineering)
        self.assertIn("训练任务", engineering)
        self.assertIn("数据导入", engineering)


if __name__ == "__main__":
    unittest.main()
