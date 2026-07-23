use std::fmt;

use clap::ValueEnum;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, ValueEnum)]
pub enum Tool {
    Claude,
    Codex,
    Gemini,
    Kimi,
    Omp,
    All,
}

impl Tool {
    pub const ROTATION: [Tool; 6] = [
        Tool::All,
        Tool::Claude,
        Tool::Codex,
        Tool::Gemini,
        Tool::Kimi,
        Tool::Omp,
    ];

    pub fn key(self) -> &'static str {
        match self {
            Tool::Claude => "claude",
            Tool::Codex => "codex",
            Tool::Gemini => "gemini",
            Tool::Kimi => "kimi",
            Tool::Omp => "omp",
            Tool::All => "all",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            Tool::Claude => "Claude Code",
            Tool::Codex => "Codex",
            Tool::Gemini => "Gemini CLI",
            Tool::Kimi => "Kimi Code",
            Tool::Omp => "Oh My Pi",
            Tool::All => "All Tools",
        }
    }

    pub fn comparison_label(self) -> &'static str {
        match self {
            Tool::All => "All",
            _ => self.display_name(),
        }
    }

    /// Compact harness tag for narrow table cells and merged harness lists.
    pub fn short_label(self) -> &'static str {
        match self {
            Tool::Claude => "CC",
            Tool::Codex => "Cdx",
            Tool::Gemini => "GCli",
            Tool::Kimi => "KC",
            Tool::Omp => "OMP",
            Tool::All => "All",
        }
    }

    pub fn from_key(value: &str) -> Option<Self> {
        match value {
            "claude" => Some(Tool::Claude),
            "codex" => Some(Tool::Codex),
            "gemini" => Some(Tool::Gemini),
            "kimi" => Some(Tool::Kimi),
            "omp" => Some(Tool::Omp),
            "all" => Some(Tool::All),
            _ => None,
        }
    }

    pub fn is_all(self) -> bool {
        self == Tool::All
    }
}

impl fmt::Display for Tool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.key())
    }
}
