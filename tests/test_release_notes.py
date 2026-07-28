import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = REPO_ROOT / "scripts" / "generate_release_notes.py"


class ReleaseNotesTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.repo = Path(self.tempdir.name)
        self.git("init", "--quiet")
        self.git("config", "user.name", "Release Test")
        self.git("config", "user.email", "release@example.com")

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=self.repo,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()

    def commit(self, subject: str, content: str) -> None:
        (self.repo / "change.txt").write_text(content)
        self.git("add", "change.txt")
        self.git("commit", "--quiet", "-m", subject)

    def test_notes_summarize_commits_since_previous_semantic_version(self):
        self.commit("Initial release", "initial")
        self.git("tag", "v3.1.1")
        self.commit("Report integrity progress", "progress")
        self.commit("Expire idle memory cache", "cache")
        self.git("tag", "v3.1.2")

        result = subprocess.run(
            [
                sys.executable,
                str(GENERATOR),
                "v3.1.2",
                "--repository-url",
                "https://github.com/example/ai-usage",
            ],
            cwd=self.repo,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("## Changes", result.stdout)
        self.assertIn("Report integrity progress", result.stdout)
        self.assertIn("Expire idle memory cache", result.stdout)
        self.assertNotIn("Initial release", result.stdout)
        self.assertIn("/compare/v3.1.1...v3.1.2", result.stdout)
        self.assertRegex(result.stdout, r"/commit/[0-9a-f]{40}")

    def test_release_workflow_publishes_generated_notes_before_assets(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text()

        self.assertIn("release_notes:", workflow)
        self.assertIn("fetch-depth: 0", workflow)
        self.assertIn("python3 scripts/generate_release_notes.py", workflow)
        self.assertIn("body_path: release-notes.md", workflow)
        self.assertIn("needs: release_notes", workflow)


if __name__ == "__main__":
    unittest.main()
