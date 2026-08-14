from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" if (ROOT / "web").is_dir() else ROOT
HTML_PATH = WEB / "index.html" if (WEB / "index.html").is_file() else ROOT / "index.current.html"
CONSOLE_SCRIPT = WEB / "console.js" if (WEB / "console.js").is_file() else ROOT / "console.js"
APP_SCRIPT = WEB / "app.js" if (WEB / "app.js").is_file() else ROOT / "app.current.js"


class ConsoleUiTests(unittest.TestCase):
    def test_console_presents_generic_visual_workflow(self):
        html = HTML_PATH.read_text(encoding="utf-8")
        self.assertIn("低空遥感智能分析平台", html)
        self.assertIn("配置识别目标", html)
        self.assertNotIn("停车设施 mask", html)
        self.assertNotIn("蒸馏流水线", html)

    def test_console_commands_map_to_supported_backend_actions(self):
        html = HTML_PATH.read_text(encoding="utf-8")
        for action in ("p1_all", "p1_infer", "p1_coords", "p1_aggregate", "p2_index", "p3_export", "p3_audit"):
            self.assertIn(f'data-job="{action}"', html)
        script = CONSOLE_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("/api/gemma/generate-config", script)
        self.assertIn("/api/gemma/apply-config", script)
        self.assertIn("/api/gemma/auto/start", script)
        self.assertIn("/api/runs/select", script)
        self.assertIn("sam3.text_prompt = prompts[0]", script)

    def test_console_has_accessible_task_dialog_and_live_log(self):
        html = HTML_PATH.read_text(encoding="utf-8")
        self.assertIn('aria-labelledby="taskDialogTitle"', html)
        self.assertIn('role="log" aria-live="polite"', html)
        self.assertIn('aria-label="关闭"', html)

    def test_shared_shell_uses_platform_brand_and_business_page_names(self):
        script = APP_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("低空遥感智能分析平台", script)
        for page_name in ("结果审核", "分析报告", "模型验证", "数据管理"):
            self.assertIn(page_name, script)
        self.assertNotIn("自动化标注训练系统 — 共享前端逻辑", script)


if __name__ == "__main__":
    unittest.main()
