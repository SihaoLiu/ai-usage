#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:
    raise SystemExit("Missing Python dependency: install Pillow to render docs/assets/demo.png") from exc

try:
    from wcwidth import wcwidth
except ImportError:
    wcwidth = None


DEFAULT_FG = (238, 238, 238)
BG = (0, 0, 0)


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
ANSI_RE = re.compile(r"\x1b\[([0-9;]*)m")
ANSI_ANY_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


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


def run_dashboard(repo_root, columns, lines, vendor, days, timeout):
    binary = repo_root / "target" / "release" / "vibe-usage"
    if not binary.exists():
        raise SystemExit(f"Missing {binary}; run `cargo build --release` first")

    env = os.environ.copy()
    env.update(
        {
            "COLUMNS": str(columns),
            "LINES": str(lines),
            "TERM": "xterm-256color",
        }
    )

    cmd = [
        str(binary),
        "--once",
        "--vendor",
        vendor,
        "--days",
        str(days),
    ]
    result = subprocess.run(
        cmd,
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)

    return result.stdout


def parse_sgr(params, fg, bold):
    if not params:
        return DEFAULT_FG, False

    values = [0 if part == "" else int(part) for part in params.split(";")]
    i = 0
    while i < len(values):
        code = values[i]
        if code == 0:
            fg = DEFAULT_FG
            bold = False
        elif code == 1:
            bold = True
        elif code == 22:
            bold = False
        elif code == 39:
            fg = DEFAULT_FG
        elif 30 <= code <= 37:
            fg = PALETTE[code - 30]
        elif 90 <= code <= 97:
            fg = PALETTE[8 + code - 90]
        elif code == 38 and i + 2 < len(values) and values[i + 1] == 5:
            color_idx = values[i + 2]
            if 0 <= color_idx < len(PALETTE):
                fg = PALETTE[color_idx]
            i += 2
        i += 1

    return fg, bold


def parse_ansi_line(line):
    spans = []
    fg = DEFAULT_FG
    bold = False
    pos = 0

    while pos < len(line):
        match = ANSI_RE.search(line, pos)
        if not match:
            text = ANSI_ANY_RE.sub("", line[pos:])
            if text:
                spans.append((text, fg, bold))
            break

        text = ANSI_ANY_RE.sub("", line[pos : match.start()])
        if text:
            spans.append((text, fg, bold))
        fg, bold = parse_sgr(match.group(1), fg, bold)
        pos = match.end()

    return spans


def visible_len(line):
    return cell_width(ANSI_ANY_RE.sub("", line))


def char_cell_width(char):
    if wcwidth is None:
        return 0 if ord(char) < 32 else 1

    width = wcwidth(char)
    return max(width, 0)


def cell_width(text):
    return sum(char_cell_width(char) for char in text)


def draw_cells(draw, origin_x, origin_y, text, font, fill, char_width):
    x_cols = 0
    for char in text:
        width = char_cell_width(char)
        if width == 0:
            continue
        draw.text((origin_x + x_cols * char_width, origin_y), char, font=font, fill=fill)
        x_cols += width
    return x_cols


def render_png(ansi_text, output, font_path, font_size, padding):
    normal = ImageFont.truetype(str(font_path), font_size)
    bold_path = Path(str(font_path).replace(".ttf", "-Bold.ttf"))
    bold = ImageFont.truetype(str(bold_path), font_size) if bold_path.exists() else normal

    probe = Image.new("RGB", (1, 1), BG)
    draw = ImageDraw.Draw(probe)
    char_box = draw.textbbox((0, 0), "M", font=normal)
    line_box = draw.textbbox((0, 0), "Hg|", font=normal)
    char_width = char_box[2] - char_box[0]
    line_height = int((line_box[3] - line_box[1]) * 1.35)

    lines = ansi_text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    max_cols = max((visible_len(line) for line in lines), default=1)
    width = padding * 2 + max_cols * char_width
    height = padding * 2 + max(1, len(lines)) * line_height

    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)

    y = padding
    for line in lines:
        x_cols = 0
        for text, fg, is_bold in parse_ansi_line(line):
            font = bold if is_bold else normal
            x_cols += draw_cells(draw, padding + x_cols * char_width, y, text, font, fg, char_width)
        y += line_height

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return width, height, len(lines)


def main():
    parser = argparse.ArgumentParser(description="Render the current dashboard to docs/assets/demo.png")
    parser.add_argument("--output", default="docs/assets/demo.png")
    parser.add_argument("--columns", type=int, default=160)
    parser.add_argument("--lines", type=int, default=80)
    parser.add_argument("--vendor", default="all")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--font")
    parser.add_argument("--font-size", type=int, default=12)
    parser.add_argument("--padding", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    repo_root = find_repo_root()
    output = (repo_root / args.output).resolve()
    font_path = find_font(repo_root, args.font)
    ansi_text = run_dashboard(repo_root, args.columns, args.lines, args.vendor, args.days, args.timeout)
    width, height, line_count = render_png(ansi_text, output, font_path, args.font_size, args.padding)

    rel_output = output.relative_to(repo_root)
    print(f"Rendered {rel_output} ({width}x{height}, {line_count} lines)")


if __name__ == "__main__":
    main()
