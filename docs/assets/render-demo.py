#!/usr/bin/env python3
import argparse
import codecs
import errno
import fcntl
import os
import pty
import select
import signal
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:
    raise SystemExit("Missing Python dependency: install Pillow to render docs/assets/ai-usage.gif") from exc

try:
    from wcwidth import wcwidth
except ImportError:
    wcwidth = None


DEFAULT_FG = (238, 238, 238)
BG = (0, 0, 0)
KEY_RIGHT = "\x1b[C"
KEY_LEFT = "\x1b[D"
KEY_PAGE_UP = "\x1b[5~"
KEY_PAGE_DOWN = "\x1b[6~"
KEY_PLUS = "+"
KEY_MINUS = "-"
KEY_HELP = "h\r"
KEY_CTRL_C = "\x03"
READY_TIMEOUT_SECONDS = 30.0


def xterm_palette():
    colors = [
        (0, 0, 0),
        (128, 0, 0),
        (0, 128, 0),
        (128, 128, 0),
        (0, 0, 128),
        (128, 0, 128),
        (0, 128, 128),
        (192, 192, 192),
        (128, 128, 128),
        (255, 0, 0),
        (0, 255, 0),
        (255, 255, 0),
        (0, 0, 255),
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 255),
    ]

    levels = [0, 95, 135, 175, 215, 255]
    for r in levels:
        for g in levels:
            for b in levels:
                colors.append((r, g, b))

    for i in range(24):
        level = 8 + i * 10
        colors.append((level, level, level))

    return colors


PALETTE = xterm_palette()


@dataclass(frozen=True)
class InputEvent:
    at: float
    data: str


@dataclass
class Cell:
    char: str = " "
    fg: tuple[int, int, int] = DEFAULT_FG
    bg: tuple[int, int, int] = BG
    bold: bool = False


@dataclass
class ScreenLine:
    text: str
    cells: list[Cell]


class TerminalScreen:
    def __init__(self, columns, rows):
        self.columns = columns
        self.rows = rows
        self.cursor_col = 0
        self.cursor_row = 0
        self.saved_cursor = (0, 0)
        self.fg = DEFAULT_FG
        self.bg = BG
        self.bold = False
        self.pending = ""
        self.pending_wrap = False
        self.utf8_decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.cells = [[self.blank_cell() for _ in range(columns)] for _ in range(rows)]

    def blank_cell(self):
        return Cell(" ", self.fg, self.bg, self.bold)

    def feed(self, data):
        if isinstance(data, bytes):
            text = self.utf8_decoder.decode(data)
        else:
            text = data

        text = self.pending + text
        self.pending = ""
        i = 0
        while i < len(text):
            char = text[i]
            if char == "\x1b":
                parsed = self.consume_escape(text, i)
                if parsed is None:
                    self.pending = text[i:]
                    break
                i = parsed
                continue
            if char == "\r":
                self.cursor_col = 0
                self.pending_wrap = False
            elif char == "\n":
                self.newline()
            elif char == "\b":
                self.cursor_col = max(0, self.cursor_col - 1)
                self.pending_wrap = False
            elif char == "\t":
                spaces = 8 - (self.cursor_col % 8)
                for _ in range(spaces):
                    self.put_char(" ")
            elif ord(char) >= 32:
                self.put_char(char)
            i += 1

    def consume_escape(self, text, start):
        if start + 1 >= len(text):
            return None

        introducer = text[start + 1]
        if introducer != "[":
            if introducer in "78":
                if introducer == "7":
                    self.saved_cursor = (self.cursor_row, self.cursor_col)
                else:
                    self.cursor_row, self.cursor_col = self.saved_cursor
                    self.clamp_cursor()
                return start + 2
            return start + 2

        end = start + 2
        while end < len(text):
            final = text[end]
            if "@" <= final <= "~":
                self.handle_csi(text[start + 2 : end], final)
                return end + 1
            end += 1
        return None

    def handle_csi(self, body, final):
        if final == "m":
            self.fg, self.bg, self.bold = parse_sgr(body, self.fg, self.bg, self.bold)
            return
        if final in "hl":
            return

        values = self.csi_numbers(body)
        first = values[0] if values else 0
        if final in "Hf":
            row = (values[0] if len(values) >= 1 and values[0] else 1) - 1
            col = (values[1] if len(values) >= 2 and values[1] else 1) - 1
            self.cursor_row = row
            self.cursor_col = col
            self.clamp_cursor()
        elif final == "A":
            self.cursor_row -= first or 1
            self.clamp_cursor()
        elif final == "B":
            self.cursor_row += first or 1
            self.clamp_cursor()
        elif final == "C":
            self.cursor_col += first or 1
            self.clamp_cursor()
        elif final == "D":
            self.cursor_col -= first or 1
            self.clamp_cursor()
        elif final == "G":
            self.cursor_col = (first or 1) - 1
            self.clamp_cursor()
        elif final == "d":
            self.cursor_row = (first or 1) - 1
            self.clamp_cursor()
        elif final == "J":
            self.clear_display(first)
        elif final == "K":
            self.clear_line(first)
        elif final == "s":
            self.saved_cursor = (self.cursor_row, self.cursor_col)
        elif final == "u":
            self.cursor_row, self.cursor_col = self.saved_cursor
            self.clamp_cursor()
        self.pending_wrap = False

    def csi_numbers(self, body):
        body = body.lstrip("?=>")
        if not body:
            return [0]
        values = []
        for part in body.split(";"):
            if part == "":
                values.append(0)
                continue
            try:
                values.append(int(part))
            except ValueError:
                values.append(0)
        return values

    def clear_display(self, mode):
        if mode in (2, 3):
            self.cells = [[self.blank_cell() for _ in range(self.columns)] for _ in range(self.rows)]
        elif mode == 1:
            for row in range(0, self.cursor_row):
                self.cells[row] = [self.blank_cell() for _ in range(self.columns)]
            for col in range(0, self.cursor_col + 1):
                self.cells[self.cursor_row][col] = self.blank_cell()
        else:
            for col in range(self.cursor_col, self.columns):
                self.cells[self.cursor_row][col] = self.blank_cell()
            for row in range(self.cursor_row + 1, self.rows):
                self.cells[row] = [self.blank_cell() for _ in range(self.columns)]

    def clear_line(self, mode):
        if mode == 1:
            start, end = 0, self.cursor_col + 1
        elif mode == 2:
            start, end = 0, self.columns
        else:
            start, end = self.cursor_col, self.columns
        for col in range(start, end):
            self.cells[self.cursor_row][col] = self.blank_cell()

    def put_char(self, char):
        if self.pending_wrap:
            self.newline()

        width = char_cell_width(char)
        if width <= 0:
            return
        if width > self.columns:
            return

        if self.cursor_col + width > self.columns:
            self.newline()

        self.cells[self.cursor_row][self.cursor_col] = Cell(char, self.fg, self.bg, self.bold)
        for col in range(self.cursor_col + 1, min(self.columns, self.cursor_col + width)):
            self.cells[self.cursor_row][col] = Cell(" ", self.fg, self.bg, self.bold)

        if self.cursor_col + width >= self.columns:
            self.cursor_col = self.columns - 1
            self.pending_wrap = True
        else:
            self.cursor_col += width

    def newline(self):
        self.pending_wrap = False
        self.cursor_row += 1
        if self.cursor_row >= self.rows:
            self.cells.pop(0)
            self.cells.append([self.blank_cell() for _ in range(self.columns)])
            self.cursor_row = self.rows - 1

    def clamp_cursor(self):
        self.cursor_row = min(max(self.cursor_row, 0), self.rows - 1)
        self.cursor_col = min(max(self.cursor_col, 0), self.columns - 1)
        self.pending_wrap = False

    def has_visible_content(self):
        return any(cell.char != " " for row in self.cells for cell in row)

    def render_lines(self):
        return [ScreenLine("".join(cell.char for cell in row), row[:]) for row in self.cells]


def find_repo_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "Cargo.toml").exists():
            return parent

    raise SystemExit("Could not find repository root")


def find_font(repo_root, name):
    if name:
        path = Path(name).expanduser()
        if path.exists():
            return path
        raise SystemExit(f"Font not found: {path}")

    candidates = [
        "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf",
        "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/google-noto/NotoSansMono-Regular.ttf",
        "/usr/share/fonts/liberation-mono/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path

    raise SystemExit("No monospace font found; pass --font PATH")


def load_braille_font(font_size):
    path = Path("/usr/share/fonts/google-noto/NotoSansSymbols2-Regular.ttf")
    if not path.exists():
        raise SystemExit("No font with Braille glyphs found for chart rendering")
    return ImageFont.truetype(str(path), max(1, round(font_size * 10 / 12)))


def font_for_cell(char, normal, bold_font, braille, *, bold):
    if "\u2800" <= char <= "\u28ff":
        return braille
    return bold_font if bold else normal


def parse_sgr(params, fg, bg, bold):
    if not params:
        return DEFAULT_FG, BG, False

    values = [0 if part == "" else int(part) for part in params.split(";")]
    i = 0
    while i < len(values):
        code = values[i]
        if code == 0:
            fg = DEFAULT_FG
            bg = BG
            bold = False
        elif code == 1:
            bold = True
        elif code == 22:
            bold = False
        elif code == 39:
            fg = DEFAULT_FG
        elif code == 49:
            bg = BG
        elif 30 <= code <= 37:
            fg = PALETTE[code - 30]
        elif 90 <= code <= 97:
            fg = PALETTE[8 + code - 90]
        elif 40 <= code <= 47:
            bg = PALETTE[code - 40]
        elif 100 <= code <= 107:
            bg = PALETTE[8 + code - 100]
        elif code == 38 and i + 2 < len(values) and values[i + 1] == 5:
            color_idx = values[i + 2]
            if 0 <= color_idx < len(PALETTE):
                fg = PALETTE[color_idx]
            i += 2
        elif code == 48 and i + 2 < len(values) and values[i + 1] == 5:
            color_idx = values[i + 2]
            if 0 <= color_idx < len(PALETTE):
                bg = PALETTE[color_idx]
            i += 2
        i += 1

    return fg, bg, bold


def char_cell_width(char):
    if wcwidth is None:
        return 0 if ord(char) < 32 else 1

    width = wcwidth(char)
    return max(width, 0)


def load_fonts(font_path, font_size):
    normal = ImageFont.truetype(str(font_path), font_size)
    bold_path = Path(str(font_path).replace(".ttf", "-Bold.ttf"))
    bold = ImageFont.truetype(str(bold_path), font_size) if bold_path.exists() else normal
    return normal, bold


def font_metrics(normal):
    probe = Image.new("RGB", (1, 1), BG)
    draw = ImageDraw.Draw(probe)
    char_box = draw.textbbox((0, 0), "M", font=normal)
    line_box = draw.textbbox((0, 0), "Hg|", font=normal)
    char_width = char_box[2] - char_box[0]
    line_height = int((line_box[3] - line_box[1]) * 1.35)
    return char_width, line_height


def render_screen_image(screen, font_path, font_size, padding):
    normal, bold = load_fonts(font_path, font_size)
    braille = load_braille_font(font_size)
    char_width, line_height = font_metrics(normal)
    width = padding * 2 + screen.columns * char_width
    height = padding * 2 + screen.rows * line_height
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)

    for row_index, line in enumerate(screen.render_lines()):
        y = padding + row_index * line_height
        for col_index, cell in enumerate(line.cells):
            x = padding + col_index * char_width
            if cell.bg != BG:
                draw.rectangle(
                    (x, y, x + char_width - 1, y + line_height - 1),
                    fill=cell.bg,
                )
            if cell.char == " ":
                continue
            font = font_for_cell(cell.char, normal, bold, braille, bold=cell.bold)
            draw.text(
                (x, y),
                cell.char,
                font=font,
                fill=cell.fg,
            )

    return image


def build_demo_events(duration=5.0, step_interval=0.1):
    events = []
    at = 0.8

    for key, count in [
        (KEY_RIGHT, 15),
        (KEY_LEFT, 15),
    ]:
        for _ in range(count):
            events.append(InputEvent(round(at, 3), key))
            at += step_interval

    events.append(InputEvent(round(at, 3), KEY_HELP))
    events.append(InputEvent(duration, KEY_CTRL_C))
    return sorted(events, key=lambda event: event.at)


def set_pty_size(fd, columns, rows):
    winsize = struct.pack("HHHH", rows, columns, 0, 0)
    fcntl.ioctl(fd, termios_tiocswinsz(), winsize)


def termios_tiocswinsz():
    import termios

    return termios.TIOCSWINSZ


def monitor_environment(columns, rows):
    env = os.environ.copy()
    env.pop("NO_COLOR", None)
    env.update(
        {
            "COLUMNS": str(columns),
            "LINES": str(rows),
            "TERM": "xterm-256color",
        }
    )
    return env


def spawn_monitor(repo_root, columns, rows, vendor, days):
    binary = repo_root / "target" / "release" / "ai-usage"
    if not binary.exists():
        raise SystemExit(f"Missing {binary}; run `cargo build --release` first")

    pid, master_fd = pty.fork()
    if pid == 0:
        env = monitor_environment(columns, rows)
        os.chdir(repo_root)
        cmd = [str(binary), "--tool", vendor, "--days", str(days)]
        try:
            os.execvpe(cmd[0], cmd, env)
        except OSError as exc:
            print(f"exec failed: {exc}", file=sys.stderr)
            os._exit(127)

    set_pty_size(master_fd, columns, rows)
    flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
    fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    return pid, master_fd


def read_available(fd, screen):
    while True:
        try:
            chunk = os.read(fd, 65536)
        except BlockingIOError:
            return True
        except OSError as exc:
            if exc.errno in (errno.EIO, errno.EBADF):
                return False
            raise
        if not chunk:
            return False
        screen.feed(chunk)


def dashboard_ready(screen):
    content = "\n".join(line.text for line in screen.render_lines())
    return (
        "Usage / API Cost" in content
        or "No usage data found from any tool." in content
    )


def wait_for_child(pid, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        waited, _status = os.waitpid(pid, os.WNOHANG)
        if waited == pid:
            return
        time.sleep(0.05)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    os.waitpid(pid, 0)


def capture_demo_frames(repo_root, args, font_path):
    screen = TerminalScreen(args.columns, args.lines)
    events = build_demo_events(args.duration, args.key_interval)
    frames = []
    frame_period = 1.0 / args.fps
    next_frame_at = 0.0
    event_index = 0
    pid, master_fd = spawn_monitor(repo_root, args.columns, args.lines, args.vendor, args.days)
    startup_started = time.monotonic()
    start = None

    try:
        while True:
            monitor_open = read_available(master_fd, screen)
            now = time.monotonic()
            if start is None:
                if dashboard_ready(screen):
                    start = now
                elif not monitor_open:
                    raise RuntimeError("monitor exited before dashboard became ready")
                elif now - startup_started >= READY_TIMEOUT_SECONDS:
                    raise RuntimeError("dashboard did not finish loading before the recording timeout")
                else:
                    select.select([master_fd], [], [], 0.05)
                    continue

            elapsed = now - start

            while event_index < len(events) and events[event_index].at <= elapsed:
                os.write(master_fd, events[event_index].data.encode("ascii"))
                event_index += 1

            while next_frame_at <= elapsed and next_frame_at <= args.duration:
                if screen.has_visible_content():
                    frames.append(render_screen_image(screen, font_path, args.font_size, args.padding))
                next_frame_at += frame_period

            if elapsed >= args.duration:
                break

            next_event_at = events[event_index].at if event_index < len(events) else args.duration
            next_wakeup = min(next_frame_at, next_event_at, args.duration)
            timeout = max(0.0, min(0.05, next_wakeup - elapsed))
            select.select([master_fd], [], [], timeout)
    finally:
        try:
            os.write(master_fd, KEY_CTRL_C.encode("ascii"))
        except OSError:
            pass
        read_available(master_fd, screen)
        os.close(master_fd)
        wait_for_child(pid)

    if not frames:
        frames.append(render_screen_image(screen, font_path, args.font_size, args.padding))
    return frames


def frame_duration_ms(fps, speed):
    if fps <= 0:
        raise ValueError("fps must be greater than zero")
    if speed <= 0:
        raise ValueError("speed must be greater than zero")
    return max(1, round(1000 / (fps * speed)))


def save_gif(frames, output, fps, speed):
    output.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = frame_duration_ms(fps, speed)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Record the monitor dashboard to docs/assets/ai-usage.gif")
    parser.add_argument("--output", default="docs/assets/ai-usage.gif")
    parser.add_argument("--columns", type=int, default=240)
    parser.add_argument("--lines", type=int, default=54)
    parser.add_argument("--vendor", default="all")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--font")
    parser.add_argument("--font-size", type=int, default=16)
    parser.add_argument("--padding", type=int, default=8)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--key-interval", type=float, default=0.1)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    repo_root = find_repo_root()
    output = (repo_root / args.output).resolve()
    font_path = find_font(repo_root, args.font)
    frames = capture_demo_frames(repo_root, args, font_path)
    save_gif(frames, output, args.fps, args.speed)

    rel_output = output.relative_to(repo_root)
    width, height = frames[0].size
    print(f"Rendered {rel_output} ({width}x{height}, {len(frames)} frames, {args.speed:g}x speed)")


if __name__ == "__main__":
    main()
