use std::path::PathBuf;
use std::process::Command;

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn fixture_dir() -> PathBuf {
    project_root().join("tests/fixtures")
}

fn reference_dir() -> PathBuf {
    project_root().join("tests/snapshots/reference")
}

fn run_binary(args: &[&str], cols: u16, lines: u16) -> String {
    let binary = project_root().join("target/release/vibe-usage");
    if !binary.exists() {
        panic!("Release binary not found. Run: cargo build --release");
    }

    // Touch fixture files so mtime filter passes
    let fixture_path = fixture_dir();
    let _ = Command::new("find")
        .args([fixture_path.to_str().unwrap(), "-type", "f", "-exec", "touch", "{}", "+"])
        .output();

    let output = Command::new(&binary)
        .args(args)
        .env("CLAUDE_CONFIG_DIR", fixture_dir().join("claude"))
        .env("CODEX_CONFIG_DIR", fixture_dir().join("codex"))
        .env("GEMINI_CONFIG_DIR", fixture_dir().join("gemini"))
        .env("COLUMNS", cols.to_string())
        .env("LINES", lines.to_string())
        .output()
        .expect("Failed to run binary");

    String::from_utf8_lossy(&output.stdout).to_string()
}

/// Extract the table portion of output (lines before "Last updated" or chart data).
fn extract_table(output: &str) -> String {
    let mut table_lines = Vec::new();
    for line in output.lines() {
        if line.starts_with("Last updated:") || line.contains("Token Consumption") {
            break;
        }
        // Skip lines with volatile timestamps
        let clean_check = strip_ansi(line);
        if clean_check.starts_with("Updated:") {
            continue;
        }
        // Strip ANSI codes for comparison
        let clean = strip_ansi(line);
        table_lines.push(clean);
    }
    table_lines.join("\n")
}

fn strip_ansi(s: &str) -> String {
    let mut result = String::new();
    let mut in_escape = false;
    for c in s.chars() {
        if c == '\x1b' {
            in_escape = true;
        } else if in_escape {
            if c.is_ascii_alphabetic() {
                in_escape = false;
            }
        } else {
            result.push(c);
        }
    }
    result
}

fn load_reference(name: &str) -> String {
    let path = reference_dir().join(format!("{}.txt", name));
    std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("Reference snapshot not found: {}", path.display()))
}

#[test]
fn snapshot_claude_full() {
    let rust_output = run_binary(&["--once", "--days", "3", "--vendor", "claude"], 200, 60);
    let reference = load_reference("claude-full");
    assert_eq!(
        extract_table(&rust_output),
        extract_table(&reference),
        "Claude full table mismatch"
    );
}

#[test]
fn snapshot_codex_full() {
    let rust_output = run_binary(&["--once", "--days", "3", "--vendor", "codex"], 200, 60);
    let reference = load_reference("codex-full");
    assert_eq!(
        extract_table(&rust_output),
        extract_table(&reference),
        "Codex full table mismatch"
    );
}

#[test]
fn snapshot_gemini_full() {
    let rust_output = run_binary(&["--once", "--days", "3", "--vendor", "gemini"], 200, 60);
    let reference = load_reference("gemini-full");
    assert_eq!(
        extract_table(&rust_output),
        extract_table(&reference),
        "Gemini full table mismatch"
    );
}

#[test]
fn snapshot_all_full() {
    let rust_output = run_binary(&["--once", "--days", "3"], 200, 60);
    let reference = load_reference("all-full");
    assert_eq!(
        extract_table(&rust_output),
        extract_table(&reference),
        "All vendors full table mismatch"
    );
}

#[test]
fn snapshot_claude_compact() {
    let rust_output = run_binary(&["--once", "--days", "3", "--vendor", "claude"], 100, 40);
    let reference = load_reference("claude-compact");
    assert_eq!(
        extract_table(&rust_output),
        extract_table(&reference),
        "Claude compact table mismatch"
    );
}

#[test]
fn snapshot_all_compact() {
    let rust_output = run_binary(&["--once", "--days", "3"], 100, 40);
    let reference = load_reference("all-compact");
    assert_eq!(
        extract_table(&rust_output),
        extract_table(&reference),
        "All vendors compact table mismatch"
    );
}
