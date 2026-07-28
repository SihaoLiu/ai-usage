#!/usr/bin/env python3

import argparse
import re
import subprocess
import sys
from pathlib import Path


SEMANTIC_TAG = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "git command failed"
        raise RuntimeError(message)
    return result.stdout.strip()


def semantic_version(tag: str) -> tuple[int, int, int]:
    match = SEMANTIC_TAG.fullmatch(tag)
    if match is None:
        raise ValueError(f"release tag must match vMAJOR.MINOR.PATCH: {tag}")
    return tuple(int(part) for part in match.groups())


def previous_tag(repo: Path, current_tag: str) -> str | None:
    current = semantic_version(current_tag)
    tags = git(repo, "tag", "--merged", current_tag, "--list", "v*.*.*").splitlines()
    candidates = [
        (semantic_version(tag), tag)
        for tag in tags
        if SEMANTIC_TAG.fullmatch(tag) and semantic_version(tag) < current
    ]
    return max(candidates, default=(None, None))[1]


def markdown_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def generate_notes(repo: Path, tag: str, repository_url: str) -> str:
    semantic_version(tag)
    git(repo, "rev-parse", "--verify", f"refs/tags/{tag}")
    previous = previous_tag(repo, tag)
    revision_range = f"{previous}..{tag}" if previous else tag
    log = git(repo, "log", "--format=%H%x1f%s", revision_range)
    commits = []
    for line in log.splitlines():
        commit_hash, separator, subject = line.partition("\x1f")
        if separator:
            commits.append((commit_hash, subject))

    base_url = repository_url.removesuffix(".git").rstrip("/")
    lines = ["## Changes", ""]
    if commits:
        lines.extend(
            f"- {markdown_text(subject)} ([{commit_hash[:7]}]({base_url}/commit/{commit_hash}))"
            for commit_hash, subject in commits
        )
    else:
        lines.append("- No commits were recorded for this release.")

    if previous:
        lines.extend(
            [
                "",
                f"**Full changelog:** [{previous}...{tag}]({base_url}/compare/{previous}...{tag})",
            ]
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate release notes from Git commits")
    parser.add_argument("tag")
    parser.add_argument("--repository-url", required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        sys.stdout.write(generate_notes(args.repo, args.tag, args.repository_url))
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
