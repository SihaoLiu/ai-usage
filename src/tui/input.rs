//! Prompt input line and command history for the monitor UI.

/// Shell-like in-memory command history for the monitor prompt.
///
/// `cursor == None` means the user is editing a fresh line. `Up` saves that
/// line into `draft` and moves to the most recent entry. `Down` walks back
/// toward the draft, ending with `cursor = None` and `input_buf = draft`.
pub struct CommandHistory {
    entries: Vec<String>,
    cursor: Option<usize>,
    draft: String,
}

impl CommandHistory {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            cursor: None,
            draft: String::new(),
        }
    }

    /// Append `command` to history (unless empty or identical to the last
    /// entry) and always reset navigation state to the fresh-line position.
    pub fn record(&mut self, command: &str) {
        self.cursor = None;
        self.draft.clear();
        if command.is_empty() {
            return;
        }
        if self.entries.last().map(|s| s.as_str()) == Some(command) {
            return;
        }
        self.entries.push(command.to_string());
    }

    /// Return the previous entry to display, or `None` if there is nothing
    /// older to walk to. When stepping off the fresh line, `current_buf` is
    /// saved as the draft so `navigate_down` can restore it later.
    pub fn navigate_up(&mut self, current_buf: &str) -> Option<String> {
        if self.entries.is_empty() {
            return None;
        }
        let new_cursor = match self.cursor {
            None => {
                self.draft = current_buf.to_string();
                self.entries.len() - 1
            }
            Some(0) => return None,
            Some(n) => n - 1,
        };
        self.cursor = Some(new_cursor);
        Some(self.entries[new_cursor].clone())
    }

    /// Return the next entry to display, or the saved draft when stepping
    /// back to the fresh-line position. `None` means the cursor is already
    /// at the fresh line and nothing changes.
    pub fn navigate_down(&mut self) -> Option<String> {
        match self.cursor {
            None => None,
            Some(n) if n + 1 < self.entries.len() => {
                self.cursor = Some(n + 1);
                Some(self.entries[n + 1].clone())
            }
            Some(_) => {
                self.cursor = None;
                Some(std::mem::take(&mut self.draft))
            }
        }
    }
}

/// Editable prompt buffer with a char-aligned cursor. Drives the shell-style
/// editing behavior (insert at cursor, backspace before cursor, left/right
/// arrows moving the cursor without changing the text).
pub struct InputLine {
    buf: String,
    cursor_chars: usize,
}

impl InputLine {
    pub fn new() -> Self {
        Self {
            buf: String::new(),
            cursor_chars: 0,
        }
    }

    pub fn snapshot(&self) -> &str {
        &self.buf
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub fn char_count(&self) -> usize {
        self.buf.chars().count()
    }

    pub fn cursor_chars(&self) -> usize {
        self.cursor_chars
    }

    pub fn insert_char(&mut self, c: char) {
        let byte_pos = byte_index_for_char(&self.buf, self.cursor_chars);
        self.buf.insert(byte_pos, c);
        self.cursor_chars += 1;
    }

    /// Delete the char immediately before the cursor. Returns whether the
    /// buffer changed.
    pub fn backspace(&mut self) -> bool {
        if self.cursor_chars == 0 {
            return false;
        }
        let prev = self.cursor_chars - 1;
        let byte_pos = byte_index_for_char(&self.buf, prev);
        self.buf.remove(byte_pos);
        self.cursor_chars = prev;
        true
    }

    pub fn move_left(&mut self) -> bool {
        if self.cursor_chars == 0 {
            return false;
        }
        self.cursor_chars -= 1;
        true
    }

    pub fn move_right(&mut self) -> bool {
        if self.cursor_chars >= self.char_count() {
            return false;
        }
        self.cursor_chars += 1;
        true
    }

    /// Replace the buffer (used by history recall) and park the cursor at
    /// the end so the user can keep typing immediately.
    pub fn replace(&mut self, s: String) {
        self.cursor_chars = s.chars().count();
        self.buf = s;
    }

    pub fn clear(&mut self) {
        self.buf.clear();
        self.cursor_chars = 0;
    }
}

fn byte_index_for_char(s: &str, char_idx: usize) -> usize {
    s.char_indices()
        .nth(char_idx)
        .map(|(b, _)| b)
        .unwrap_or(s.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_line_insert_advances_cursor_and_appends_to_buffer() {
        let mut input = InputLine::new();
        input.insert_char('a');
        input.insert_char('b');
        input.insert_char('c');
        assert_eq!(input.snapshot(), "abc");
        assert_eq!(input.cursor_chars(), 3);
    }

    #[test]
    fn input_line_edits_at_cursor_position() {
        let mut input = InputLine::new();
        for c in "ac".chars() {
            input.insert_char(c);
        }
        assert!(input.move_left());
        input.insert_char('b');
        assert_eq!(input.snapshot(), "abc");
        assert!(input.backspace());
        assert_eq!(input.snapshot(), "ac");
        assert!(input.move_right());
        assert_eq!(input.cursor_chars(), 2);
        assert!(!input.move_right());
    }

    #[test]
    fn input_line_handles_multibyte_chars() {
        let mut input = InputLine::new();
        input.insert_char('x');
        for c in "ab".chars() {
            input.insert_char(c);
        }
        assert!(input.move_left());
        assert!(input.move_left());
        input.insert_char('b');
        assert_eq!(input.cursor_chars(), 2);
        assert!(input.backspace());
        assert_eq!(input.snapshot(), "xab");
    }

    #[test]
    fn history_walks_up_and_down_preserving_draft() {
        let mut history = CommandHistory::new();
        history.record("first");
        history.record("second");

        assert_eq!(history.navigate_up("draft"), Some("second".to_string()));
        assert_eq!(history.navigate_up("second"), Some("first".to_string()));
        assert_eq!(history.navigate_up("first"), None);
        assert_eq!(history.navigate_down(), Some("second".to_string()));
        assert_eq!(history.navigate_down(), Some("draft".to_string()));
        assert_eq!(history.navigate_down(), None);
    }

    #[test]
    fn history_skips_empty_and_duplicate_entries() {
        let mut history = CommandHistory::new();
        history.record("");
        assert_eq!(history.navigate_up(""), None);
        history.record("cmd");
        history.record("cmd");
        assert_eq!(history.navigate_up(""), Some("cmd".to_string()));
        assert_eq!(history.navigate_up("cmd"), None);
    }
}
