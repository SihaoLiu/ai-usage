import importlib.util
import sys
import tempfile
import textwrap
import time
import unittest
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch


def load_render_demo():
    script = Path(__file__).resolve().parents[1] / "docs" / "assets" / "render-demo.py"
    spec = importlib.util.spec_from_file_location("render_demo", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def fake_monitor_repo(body):
    temp_root = Path.cwd() / "temp"
    temp_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="render-demo-test-", dir=temp_root) as directory:
        repo_root = Path(directory)
        binary = repo_root / "target" / "release" / "ai-usage"
        binary.parent.mkdir(parents=True)
        binary.write_text(
            "#!/usr/bin/env python3\n" + textwrap.dedent(body),
            encoding="utf-8",
        )
        binary.chmod(0o755)
        yield repo_root


def recording_args(duration=0.05):
    return Namespace(
        columns=80,
        lines=4,
        vendor="all",
        days=3,
        duration=duration,
        fps=20.0,
        key_interval=0.1,
        font_size=12,
        padding=2,
    )


class RenderDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.render_demo = load_render_demo()

    def test_default_output_is_readme_gif(self):
        args = self.render_demo.build_parser().parse_args([])
        self.assertEqual(args.output, "docs/assets/ai-usage.gif")
        self.assertEqual(args.duration, 5.0)
        self.assertEqual(args.speed, 1.0)
        self.assertEqual(args.key_interval, 0.1)
        self.assertEqual(args.columns, 240)
        self.assertEqual(args.font_size, 16)
        self.assertEqual(args.padding, 8)
        self.assertEqual(args.fps, 12.0)

    def test_frame_duration_uses_playback_speed(self):
        self.assertEqual(self.render_demo.frame_duration_ms(fps=8.0, speed=3.0), 42)

    def test_braille_cells_use_the_dedicated_symbol_font(self):
        normal, bold = self.render_demo.load_fonts(
            self.render_demo.find_font(Path.cwd(), None), 12
        )
        braille = self.render_demo.load_braille_font(12)

        self.assertIs(
            self.render_demo.font_for_cell("⣿", normal, bold, braille, bold=False),
            braille,
        )
        self.assertIs(
            self.render_demo.font_for_cell("A", normal, bold, braille, bold=False),
            normal,
        )
        self.assertNotEqual(
            bytes(braille.getmask("⣿")),
            bytes(braille.getmask("\ufffd")),
        )

    def test_demo_events_match_viewport_navigation_sequence(self):
        events = self.render_demo.build_demo_events(duration=5.0, step_interval=0.1)

        expected_sequence = (
            [self.render_demo.KEY_RIGHT] * 15
            + [self.render_demo.KEY_LEFT] * 15
            + [self.render_demo.KEY_HELP]
            + [self.render_demo.KEY_CTRL_C]
        )

        self.assertEqual([event.data for event in events], expected_sequence)
        self.assertEqual(self.render_demo.KEY_HELP, "h\r")
        self.assertEqual(events[-1].data, self.render_demo.KEY_CTRL_C)
        self.assertAlmostEqual(events[-1].at, 5.0)

    def test_recording_environment_enables_terminal_colors(self):
        environment = self.render_demo.monitor_environment(columns=196, rows=54)

        self.assertEqual(environment["TERM"], "xterm-256color")
        self.assertNotIn("NO_COLOR", environment)

    def test_dashboard_ready_requires_a_result_view(self):
        screen = self.render_demo.TerminalScreen(columns=80, rows=4)

        screen.feed("Loading usage history...")
        self.assertFalse(self.render_demo.dashboard_ready(screen))

        screen.feed("\x1b[2J\x1b[HUsage / API Cost (Vendor / Model / Harness)")
        self.assertTrue(self.render_demo.dashboard_ready(screen))

        screen.feed("\x1b[2J\x1b[HNo usage data found from any tool.")
        self.assertTrue(self.render_demo.dashboard_ready(screen))

    def test_capture_starts_after_dashboard_is_ready(self):
        monitor = """
            import sys
            import time

            sys.stdout.write("\\x1b[2J\\x1b[HLoading usage history...")
            sys.stdout.flush()
            time.sleep(0.1)
            sys.stdout.write("\\x1b[2J\\x1b[HUsage / API Cost (Vendor / Model / Harness)")
            sys.stdout.flush()
            time.sleep(0.3)
        """
        rendered_screens = []

        def record_screen(screen, _font_path, _font_size, _padding):
            rendered_screens.append("\n".join(line.text for line in screen.render_lines()))
            return self.render_demo.Image.new("RGB", (1, 1))

        with fake_monitor_repo(monitor) as repo_root:
            with patch.object(self.render_demo, "render_screen_image", side_effect=record_screen):
                frames = self.render_demo.capture_demo_frames(
                    repo_root,
                    recording_args(),
                    Path("unused-font"),
                )

        self.assertTrue(frames)
        self.assertTrue(rendered_screens)
        self.assertTrue(
            all("Usage / API Cost" in screen for screen in rendered_screens),
            rendered_screens,
        )

    def test_capture_fails_promptly_when_monitor_exits_before_ready(self):
        monitor = """
            import sys

            sys.stdout.write("Loading usage history...")
            sys.stdout.flush()
        """

        with fake_monitor_repo(monitor) as repo_root:
            started = time.monotonic()
            with patch.object(self.render_demo, "READY_TIMEOUT_SECONDS", 1.0):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "monitor exited before dashboard became ready",
                ):
                    self.render_demo.capture_demo_frames(
                        repo_root,
                        recording_args(),
                        Path("unused-font"),
                    )

        self.assertLess(time.monotonic() - started, 0.5)

    def test_terminal_screen_handles_clear_line_and_cursor_home(self):
        screen = self.render_demo.TerminalScreen(columns=10, rows=3)

        screen.feed("hello\r\x1b[Kbye")
        self.assertEqual(screen.render_lines()[0].text.rstrip(), "bye")

        screen.feed("\x1b[2J\x1b[Htop")
        self.assertEqual(screen.render_lines()[0].text.rstrip(), "top")
        self.assertEqual(screen.render_lines()[1].text.rstrip(), "")

    def test_terminal_screen_keeps_sgr_color_state(self):
        screen = self.render_demo.TerminalScreen(columns=5, rows=1)

        screen.feed("a\x1b[31mb\x1b[0mc")
        line = screen.render_lines()[0]

        self.assertEqual(line.text[:3], "abc")
        self.assertEqual(line.cells[0].fg, self.render_demo.DEFAULT_FG)
        self.assertEqual(line.cells[1].fg, self.render_demo.PALETTE[1])
        self.assertEqual(line.cells[2].fg, self.render_demo.DEFAULT_FG)

    def test_terminal_screen_keeps_sgr_background_state(self):
        screen = self.render_demo.TerminalScreen(columns=5, rows=1)

        screen.feed("a\x1b[48;5;233m \x1b[49mc")
        line = screen.render_lines()[0]

        self.assertEqual(line.cells[0].bg, self.render_demo.BG)
        self.assertEqual(line.cells[1].bg, self.render_demo.PALETTE[233])
        self.assertEqual(line.cells[2].bg, self.render_demo.BG)

    def test_render_screen_image_paints_blank_cell_background(self):
        screen = self.render_demo.TerminalScreen(columns=1, rows=1)
        screen.feed("\x1b[48;5;233m ")
        font_path = self.render_demo.find_font(Path.cwd(), None)

        image = self.render_demo.render_screen_image(
            screen,
            font_path,
            font_size=12,
            padding=2,
        )

        self.assertEqual(image.getpixel((2, 2)), self.render_demo.PALETTE[233])

    def test_terminal_screen_keeps_a_split_braille_utf8_sequence_intact(self):
        screen = self.render_demo.TerminalScreen(columns=5, rows=1)
        encoded = "⣿".encode("utf-8")

        screen.feed(encoded[:1])
        screen.feed(encoded[1:])

        line = screen.render_lines()[0]
        self.assertEqual(line.text[0], "⣿")
        self.assertNotIn("\ufffd", line.text)


if __name__ == "__main__":
    unittest.main()
