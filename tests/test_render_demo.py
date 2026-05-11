import importlib.util
import sys
import unittest
from pathlib import Path


def load_render_demo():
    script = Path(__file__).resolve().parents[1] / "docs" / "assets" / "render-demo.py"
    spec = importlib.util.spec_from_file_location("render_demo", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RenderDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.render_demo = load_render_demo()

    def test_default_output_is_readme_gif(self):
        args = self.render_demo.build_parser().parse_args([])
        self.assertEqual(args.output, "docs/assets/ai-usage.gif")
        self.assertEqual(args.speed, 3.0)

    def test_frame_duration_uses_playback_speed(self):
        self.assertEqual(self.render_demo.frame_duration_ms(fps=8.0, speed=3.0), 42)

    def test_demo_events_cover_arrows_pages_and_exit(self):
        events = self.render_demo.build_demo_events(duration=15.0, step_interval=1.0)

        right_times = [event.at for event in events if event.data == self.render_demo.KEY_RIGHT]
        left_times = [event.at for event in events if event.data == self.render_demo.KEY_LEFT]
        page_up_times = [event.at for event in events if event.data == self.render_demo.KEY_PAGE_UP]
        page_down_times = [event.at for event in events if event.data == self.render_demo.KEY_PAGE_DOWN]

        self.assertGreaterEqual(len(right_times), 4)
        self.assertGreaterEqual(len(left_times), 4)
        self.assertTrue(all(1.0 <= at < 6.0 for at in right_times))
        self.assertTrue(all(6.0 <= at < 11.0 for at in left_times))
        self.assertEqual(len(page_up_times), 1)
        self.assertEqual(len(page_down_times), 1)
        self.assertLess(page_up_times[0], page_down_times[0])
        self.assertEqual(events[-1].data, self.render_demo.KEY_CTRL_C)
        self.assertAlmostEqual(events[-1].at, 15.0)

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


if __name__ == "__main__":
    unittest.main()
