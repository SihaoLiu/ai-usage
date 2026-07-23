import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALLER = REPO_ROOT / "scripts" / "install.sh"


class InstallScriptTests(unittest.TestCase):
    def write_executable(self, path: Path, content: str) -> None:
        path.write_text(content)
        path.chmod(path.stat().st_mode | stat.S_IXUSR)

    def run_installer(self, system: str, machine: str, *args: str) -> tuple[subprocess.CompletedProcess[str], Path, Path, Path]:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        root = Path(tempdir.name)
        tools = root / "tools"
        tools.mkdir()
        curl_log = root / "curl-url"
        launch_log = root / "launch-args"
        install_dir = root / "install"

        self.write_executable(
            tools / "uname",
            "#!/bin/sh\n"
            "case \"$1\" in\n"
            "  -s) printf '%s\\n' \"$AI_USAGE_TEST_SYSTEM\" ;;\n"
            "  -m) printf '%s\\n' \"$AI_USAGE_TEST_MACHINE\" ;;\n"
            "  *) exit 64 ;;\n"
            "esac\n",
        )
        self.write_executable(
            tools / "curl",
            "#!/bin/sh\n"
            "set -eu\n"
            "output=\n"
            "url=\n"
            "while [ \"$#\" -gt 0 ]; do\n"
            "  case \"$1\" in\n"
            "    -o) output=$2; shift 2 ;;\n"
            "    -*) shift ;;\n"
            "    *) url=$1; shift ;;\n"
            "  esac\n"
            "done\n"
            "printf '%s\\n' \"$url\" > \"$AI_USAGE_TEST_CURL_LOG\"\n"
            "cat > \"$output\" <<'PAYLOAD'\n"
            "#!/bin/sh\n"
            "printf '%s\\n' \"$*\" > \"$AI_USAGE_TEST_LAUNCH_LOG\"\n"
            "PAYLOAD\n",
        )

        environment = os.environ.copy()
        environment.update(
            {
                "AI_USAGE_TEST_SYSTEM": system,
                "AI_USAGE_TEST_MACHINE": machine,
                "AI_USAGE_TEST_CURL_LOG": str(curl_log),
                "AI_USAGE_TEST_LAUNCH_LOG": str(launch_log),
                "AI_USAGE_INSTALL_DIR": str(install_dir),
                "HOME": str(root / "home"),
                "PATH": f"{tools}{os.pathsep}{environment['PATH']}",
            }
        )
        result = subprocess.run(
            ["sh", str(INSTALLER), *args],
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        return result, curl_log, launch_log, install_dir

    def test_linux_x86_64_installs_static_release_and_launches_it(self):
        result, curl_log, launch_log, install_dir = self.run_installer(
            "Linux", "x86_64", "--once"
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            curl_log.read_text().strip(),
            "https://github.com/SihaoLiu/ai-usage/releases/latest/download/ai-usage-x86_64-linux-musl",
        )
        self.assertTrue(os.access(install_dir / "ai-usage", os.X_OK))
        self.assertEqual(launch_log.read_text().strip(), "--once")

    def test_apple_silicon_selects_the_darwin_release_asset(self):
        result, curl_log, launch_log, install_dir = self.run_installer("Darwin", "arm64")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            curl_log.read_text().strip(),
            "https://github.com/SihaoLiu/ai-usage/releases/latest/download/ai-usage-aarch64-apple-darwin",
        )
        self.assertTrue(os.access(install_dir / "ai-usage", os.X_OK))
        self.assertEqual(launch_log.read_text().strip(), "")

    def test_unsupported_architecture_fails_before_downloading(self):
        result, curl_log, _launch_log, _install_dir = self.run_installer("Linux", "ppc64le")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unsupported Linux architecture", result.stderr)
        self.assertFalse(curl_log.exists())


if __name__ == "__main__":
    unittest.main()
