//! Human-facing interactive mode: a split-pane TUI over one resident world.
//!
//! Unlike the agent-facing one-shot commands (each a fresh load → run → exit),
//! `tui` keeps a single [`App`] (world + metric recorder) in memory and drives it
//! from an interactive prompt. Every read and mutation reuses the exact same
//! command dispatch as the one-shot path ([`App::run_oneshot`]), rendered into a
//! scrollback pane. Mutations stay in memory until an explicit `save`.

use crate::{build_world_cli, App, REPORT_EVERY};
use anyhow::{anyhow, bail, Result};
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Paragraph};
use ratatui::{DefaultTerminal, Frame};
use std::cell::Cell;
use std::path::Path;
use std::time::{Duration, Instant};
use views::output::Format;
use views::{
    load_sidecar, load_world, resolve_metrics_path, save_sidecar, save_world, sibling_metrics_path,
};

const ADVANCE_CHUNK: u64 = 200;
const SCROLLBACK_CAP: usize = 5000;

const COL_PROMPT: Color = Color::Cyan;
const COL_ERROR: Color = Color::Red;
const COL_INFO: Color = Color::Green;

/// Entry point dispatched from `main::run` for the `tui` subcommand. `in_path` is
/// the global `--in` flag (parsed and stripped by `main::run` before dispatch).
pub(crate) fn run_tui_cli(args: &[&str], in_path: Option<&str>) -> Result<()> {
    let (mut session, exec) = Session::from_args(args, in_path)?;
    if let Some(script) = exec {
        return session.run_headless(&script);
    }
    session.run_interactive()
}

/// What submitting one command line asks the driver to do next. Parsing, meta
/// commands, and read commands are all resolved inside [`Session::submit`] (which
/// pushes their output to the scrollback); only the multi-tick advance is handed
/// back so the interactive driver can chunk it with a live status bar + cancel.
enum Submit {
    /// Output already pushed to the scrollback; nothing else to do.
    Handled,
    /// Advance the world this many ticks (chunked + cancellable in the UI).
    Advance(u64),
    /// Exit the session.
    Quit,
}

struct Session {
    app: App,
    /// Path a bare `save` writes to (the opened `--in`; `None` after `--new`).
    save_path: Option<String>,
    scrollback: Vec<Line<'static>>,
    input: String,
    history: Vec<String>,
    /// Index into `history` while recalling; `None` = editing a fresh line.
    history_pos: Option<usize>,
    /// Wrapped output rows scrolled up from the bottom (0 = pinned to bottom).
    scroll: u16,
    /// Max scrollable offset from the last render (wrapped rows − viewport
    /// height), so key handlers can clamp `scroll` without knowing the area.
    max_scroll: Cell<u16>,
    dirty: bool,
    status: StatusLine,
    /// Set after a `quit` with unsaved changes; a second confirms.
    quit_armed: bool,
}

#[derive(Default, Clone)]
struct StatusLine {
    turn: u64,
    world_width: u32,
    population: usize,
}

impl Session {
    fn from_args(args: &[&str], in_path: Option<&str>) -> Result<(Self, Option<String>)> {
        let mut app = App {
            sim: None,
            recorder: None,
            report_every: REPORT_EVERY,
            format: Format::Text,
            scaled: false,
        };

        let in_path: Option<String> = in_path.map(str::to_string);
        let mut is_new = false;
        let mut exec: Option<String> = None;
        let mut passthrough: Vec<&str> = Vec::new();
        let mut i = 0;
        while i < args.len() {
            match args[i] {
                "--new" => {
                    is_new = true;
                    i += 1;
                }
                "--exec" => {
                    // Headless scripting seam: run `;`-separated commands, print
                    // the resulting scrollback, and exit (no terminal needed).
                    exec = Some(
                        args.get(i + 1)
                            .ok_or_else(|| anyhow!("--exec needs a script string"))?
                            .to_string(),
                    );
                    i += 2;
                }
                "--help" | "-h" => {
                    bail!(
                        "tui: interactive session over one resident world.\n\
                         usage: cli tui --in <world.bin>\n\
                         \x20      cli tui --new [--seed N] [--set k=v]... [--config P] [--scale W,POP]\n\
                         inside: any read/mutation command, plus save/help/clear/quit; changes are in-memory until `save`."
                    );
                }
                other => {
                    passthrough.push(other);
                    i += 1;
                }
            }
        }

        if is_new == in_path.is_some() {
            bail!("tui: pass exactly one of `--in <world.bin>` or `--new [build args]`");
        }

        let save_path;
        if let Some(path) = in_path {
            let sim = load_world(&path)?;
            app.scaled = sim.experiment_scaled();
            app.sim = Some(sim);
            if let Some(mp) = resolve_metrics_path(&path, None, false) {
                if Path::new(&mp).exists() {
                    let (report_every, recorder) = load_sidecar(&mp)?;
                    app.report_every = report_every;
                    if recorder.recorded_through_turn == app.sim.as_ref().unwrap().turn() {
                        app.recorder = Some(recorder);
                    } else {
                        app.start_recording()?;
                    }
                } else {
                    app.start_recording()?;
                }
            }
            save_path = Some(path);
        } else {
            // `--new`: reuse the exact one-shot constructor, then mint a sidecar
            // (the one-shot `new` does the same two steps in `main::run`).
            let mut sink: Vec<u8> = Vec::new();
            build_world_cli(&mut app, &passthrough, &mut sink)?;
            app.format = Format::Text;
            app.start_recording()?;
            save_path = None;
        }

        let mut session = Session {
            app,
            save_path,
            scrollback: Vec::new(),
            input: String::new(),
            history: Vec::new(),
            history_pos: None,
            scroll: 0,
            max_scroll: Cell::new(0),
            dirty: false,
            status: StatusLine::default(),
            quit_armed: false,
        };
        session.refresh_status();
        session.push_info(
            "interactive session — type a command, `help` for the list, `quit` to exit.",
        );
        Ok((session, exec))
    }

    // ---- scrollback helpers -------------------------------------------------

    fn push_line(&mut self, line: Line<'static>) {
        self.scrollback.push(line);
        if self.scrollback.len() > SCROLLBACK_CAP {
            let overflow = self.scrollback.len() - SCROLLBACK_CAP;
            self.scrollback.drain(0..overflow);
        }
        self.scroll = 0; // any new output pins the view to the bottom
    }

    fn push_styled(&mut self, text: &str, style: Style) {
        for raw in text.split('\n') {
            self.push_line(Line::from(Span::styled(raw.to_string(), style)));
        }
    }

    fn push_plain(&mut self, text: &str) {
        self.push_styled(text, Style::default());
    }

    fn push_info(&mut self, text: &str) {
        self.push_styled(text, Style::default().fg(COL_INFO));
    }

    fn push_error(&mut self, text: &str) {
        self.push_styled(text, Style::default().fg(COL_ERROR));
    }

    // ---- status -------------------------------------------------------------

    fn refresh_status(&mut self) {
        let Some(sim) = self.app.sim.as_ref() else {
            return;
        };
        self.status = StatusLine {
            turn: sim.turn(),
            world_width: sim.config().world_width,
            population: sim.organisms().len(),
        };
    }

    // ---- command submission (pure; no terminal) -----------------------------

    /// Handle one submitted line: echo it, then run a meta/read/mutation command.
    /// Advancing commands (`step`/`run-to`) return [`Submit::Advance`] so the
    /// caller can drive the tick loop; everything else pushes its own output.
    fn submit(&mut self, raw: &str) -> Submit {
        let line = raw.trim();
        // Echo the prompt line (unless it is the auto-run of an empty line).
        self.push_line(Line::from(vec![
            Span::styled(
                "> ",
                Style::default().fg(COL_PROMPT).add_modifier(Modifier::BOLD),
            ),
            Span::styled(line.to_string(), Style::default().fg(COL_PROMPT)),
        ]));
        if line.is_empty() {
            return Submit::Handled;
        }
        if !self.history.last().map(|h| h == line).unwrap_or(false) {
            self.history.push(line.to_string());
        }

        let mut parts = line.split_whitespace();
        let cmd = parts.next().unwrap_or("");
        let args: Vec<&str> = parts.collect();

        // Any command other than a bare `quit` disarms the quit confirmation.
        if !matches!(cmd, "quit" | "exit" | "q") {
            self.quit_armed = false;
        }

        match cmd {
            "quit" | "exit" | "q" => {
                if self.dirty && !self.quit_armed {
                    self.quit_armed = true;
                    self.push_info(
                        "unsaved changes — `save` to persist, or `quit` again to discard.",
                    );
                    Submit::Handled
                } else {
                    Submit::Quit
                }
            }
            "quit!" | "q!" => Submit::Quit,
            "help" | "?" => {
                self.push_help();
                Submit::Handled
            }
            "clear" => {
                self.scrollback.clear();
                self.scroll = 0;
                Submit::Handled
            }
            "save" => {
                self.handle_save(args.first().copied());
                Submit::Handled
            }
            "query" => {
                self.push_error(
                    "`query` is a batch mode for the one-shot CLI; just type commands here.",
                );
                Submit::Handled
            }
            "step" | "run-to" => match self.advance_target(cmd, &args) {
                Ok(0) => {
                    self.push_info("already at or past that turn.");
                    Submit::Handled
                }
                Ok(n) => Submit::Advance(n),
                Err(e) => {
                    self.push_error(&e.to_string());
                    Submit::Handled
                }
            },
            _ => {
                self.run_reused(cmd, &args);
                Submit::Handled
            }
        }
    }

    /// Dispatch a read / non-advancing mutation through the shared one-shot path
    /// into an in-memory buffer, then fold its output into the scrollback.
    fn run_reused(&mut self, cmd: &str, args: &[&str]) {
        let mut buf: Vec<u8> = Vec::new();
        match self.app.run_oneshot(cmd, args, &mut buf) {
            Ok(()) => {
                let text = String::from_utf8_lossy(&buf);
                let text = text.trim_end_matches('\n');
                if text.is_empty() {
                    self.push_info("(ok)");
                } else {
                    self.push_plain(text);
                }
                // `watch`/`bench` advance or discard; `watch` mutates the world.
                if matches!(cmd, "watch") {
                    self.dirty = true;
                    self.refresh_status();
                }
            }
            Err(e) => self.push_error(&e.to_string()),
        }
    }

    /// Parse `step [N]` / `run-to T` into a tick count relative to now.
    fn advance_target(&self, cmd: &str, args: &[&str]) -> Result<u64> {
        let turn = self.app.sim.as_ref().map(|s| s.turn()).unwrap_or(0);
        match cmd {
            "step" => {
                let n: u64 = args.first().map(|a| a.parse()).transpose()?.unwrap_or(1);
                Ok(n)
            }
            "run-to" => {
                let target: u64 = args
                    .first()
                    .ok_or_else(|| anyhow!("run-to needs a target turn"))?
                    .parse()?;
                Ok(target.saturating_sub(turn))
            }
            _ => unreachable!("advance_target called with {cmd}"),
        }
    }

    fn handle_save(&mut self, arg_path: Option<&str>) {
        let path = match arg_path
            .map(str::to_string)
            .or_else(|| self.save_path.clone())
        {
            Some(p) => p,
            None => {
                self.push_error("save: no path — this world was `--new`; use `save <path>`.");
                return;
            }
        };
        if let Err(e) = self.save_to(&path) {
            self.push_error(&format!("save failed: {e}"));
            return;
        }
        self.dirty = false;
        if self.save_path.is_none() {
            self.save_path = Some(path.clone());
        }
        self.push_info(&format!("wrote {path}"));
    }

    fn save_to(&self, path: &str) -> Result<()> {
        let sim = self
            .app
            .sim
            .as_ref()
            .ok_or_else(|| anyhow!("no world loaded"))?;
        save_world(sim, path)?;
        if let Some(recorder) = self.app.recorder.as_ref() {
            save_sidecar(self.app.report_every, recorder, &sibling_metrics_path(path))?;
        }
        Ok(())
    }

    fn push_help(&mut self) {
        let help = "\
commands (same vocabulary as the one-shot CLI):
  reads       turn state pillars eco lineage genome timeseries
              inspect ID  top FIELD [N]  hist FIELD  find EXPR  brain ID  decide ID
  advance     step [N]   run-to T        (Esc cancels a long run)
  other       watch T [--every E]        bench [N]
  session     save [path]  clear  help  quit  (quit! discards unsaved changes)
keys          Up/Down scroll output · PageUp/PageDown page · Home/End top/bottom
              Ctrl-P/Ctrl-N command history · Ctrl-C quit";
        self.push_info(help);
    }

    // ---- headless driver (for `--exec` and verification) --------------------

    fn run_headless(&mut self, script: &str) -> Result<()> {
        for stmt in script.split(';') {
            let stmt = stmt.trim();
            if stmt.is_empty() {
                continue;
            }
            match self.submit(stmt) {
                Submit::Advance(n) => {
                    self.app.advance(n)?;
                    self.dirty = true;
                    self.refresh_status();
                    self.push_info(&format!("advanced → {}", self.status.turn));
                }
                Submit::Quit => break,
                Submit::Handled => {}
            }
        }
        // Print the scrollback to stdout so a non-TTY caller can read results.
        let mut stdout = std::io::stdout().lock();
        use std::io::Write;
        for line in &self.scrollback {
            writeln!(stdout, "{}", line_to_plain(line))?;
        }
        Ok(())
    }

    // ---- interactive driver -------------------------------------------------

    fn run_interactive(&mut self) -> Result<()> {
        // `ratatui::init` enters the alternate screen + raw mode and installs a
        // panic hook that restores the terminal, so a crash never wedges it.
        let mut terminal = ratatui::init();
        let result = self.event_loop(&mut terminal);
        ratatui::restore();
        result
    }

    fn event_loop(&mut self, terminal: &mut DefaultTerminal) -> Result<()> {
        loop {
            terminal.draw(|frame| self.render(frame, None))?;
            if !event::poll(Duration::from_millis(250))? {
                continue;
            }
            let Event::Key(key) = event::read()? else {
                continue;
            };
            if key.kind != KeyEventKind::Press {
                continue;
            }
            // Ctrl combos: quit, and readline-style command history (Up/Down are
            // reserved for scrolling the output).
            if key.modifiers.contains(KeyModifiers::CONTROL) {
                match key.code {
                    KeyCode::Char('c') => return Ok(()),
                    KeyCode::Char('p') => self.recall_history(-1),
                    KeyCode::Char('n') => self.recall_history(1),
                    _ => {}
                }
                continue;
            }
            match key.code {
                KeyCode::Enter => {
                    let line = std::mem::take(&mut self.input);
                    self.history_pos = None;
                    match self.submit(&line) {
                        Submit::Quit => return Ok(()),
                        Submit::Advance(n) => self.run_advance(terminal, n)?,
                        Submit::Handled => {}
                    }
                }
                KeyCode::Char(c) => self.input.push(c),
                KeyCode::Backspace => {
                    self.input.pop();
                }
                // Arrows / PageUp/PageDown scroll the output pane (rows up from
                // the bottom), clamped to the last render's scrollable range.
                KeyCode::Up => self.scroll_by(1),
                KeyCode::Down => self.scroll = self.scroll.saturating_sub(1),
                KeyCode::PageUp => self.scroll_by(10),
                KeyCode::PageDown => self.scroll = self.scroll.saturating_sub(10),
                KeyCode::Home => self.scroll = self.max_scroll.get(),
                KeyCode::End => self.scroll = 0,
                KeyCode::Esc => self.input.clear(),
                _ => {}
            }
        }
    }

    /// Advance in chunks, repainting the status bar and honoring Esc-to-cancel,
    /// so a long `run-to` stays responsive.
    fn run_advance(&mut self, terminal: &mut DefaultTerminal, total: u64) -> Result<()> {
        let started = Instant::now();
        let mut done = 0u64;
        let mut cancelled = false;
        while done < total {
            let n = ADVANCE_CHUNK.min(total - done);
            if let Err(e) = self.app.advance(n) {
                self.push_error(&format!("advance failed: {e}"));
                break;
            }
            done += n;
            self.refresh_status();
            terminal.draw(|frame| self.render(frame, Some((done, total))))?;
            if event::poll(Duration::from_millis(0))? {
                if let Event::Key(key) = event::read()? {
                    let ctrl_c = key.modifiers.contains(KeyModifiers::CONTROL)
                        && key.code == KeyCode::Char('c');
                    if key.code == KeyCode::Esc || ctrl_c {
                        cancelled = true;
                        break;
                    }
                }
            }
        }
        self.dirty = true;
        self.refresh_status();
        let secs = started.elapsed().as_secs_f64();
        let note = if cancelled { " (cancelled)" } else { "" };
        self.push_info(&format!(
            "advanced → {} ({done} ticks{note}, {secs:.2}s)",
            self.status.turn
        ));
        Ok(())
    }

    /// Scroll the output up by `rows`, clamped to the last render's max offset
    /// so scrolling past the top doesn't accumulate dead presses.
    fn scroll_by(&mut self, rows: u16) {
        self.scroll = self.scroll.saturating_add(rows).min(self.max_scroll.get());
    }

    fn recall_history(&mut self, dir: i32) {
        if self.history.is_empty() {
            return;
        }
        let last = self.history.len() - 1;
        let next = match (self.history_pos, dir) {
            (None, -1) => Some(last),
            (Some(0), -1) => Some(0),
            (Some(p), -1) => Some(p - 1),
            (Some(p), 1) if p >= last => None,
            (Some(p), 1) => Some(p + 1),
            (None, _) => None,
            _ => self.history_pos,
        };
        self.history_pos = next;
        self.input = next.map(|p| self.history[p].clone()).unwrap_or_default();
    }

    // ---- rendering ----------------------------------------------------------

    fn render(&self, frame: &mut Frame, progress: Option<(u64, u64)>) {
        let areas = Layout::vertical([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(3),
        ])
        .split(frame.area());
        self.render_status(frame, areas[0], progress);
        self.render_output(frame, areas[1]);
        self.render_input(frame, areas[2]);
    }

    fn render_status(&self, frame: &mut Frame, area: Rect, progress: Option<(u64, u64)>) {
        let dirty = if self.dirty { " ●" } else { "" };
        let running = match progress {
            Some((done, total)) => format!("  running… {done}/{total}"),
            None => String::new(),
        };
        let text = format!(
            " {w}×{w} · t={} · pop {}{dirty}{running}",
            self.status.turn,
            self.status.population,
            w = self.status.world_width,
        );
        let bar = Paragraph::new(text).style(
            Style::default()
                .bg(Color::Rgb(30, 34, 42))
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        );
        frame.render_widget(bar, area);
    }

    fn render_output(&self, frame: &mut Frame, area: Rect) {
        let block = Block::bordered().title(" output ");
        let inner_w = area.width.saturating_sub(2) as usize;
        let inner_h = area.height.saturating_sub(2) as usize;
        // Pre-wrap every logical line to the pane width so long output never
        // overflows the right edge. Wrapping ourselves (rather than letting
        // Paragraph wrap) keeps the row math exact, so the newest line is always
        // visible and scroll counts real screen rows.
        let rows: Vec<Line> = self
            .scrollback
            .iter()
            .flat_map(|line| wrap_line(line, inner_w))
            .collect();
        let total = rows.len();
        let max_offset = total.saturating_sub(inner_h);
        self.max_scroll
            .set(max_offset.min(u16::MAX as usize) as u16);
        let top = max_offset.saturating_sub(self.scroll as usize);
        let end = (top + inner_h).min(total);
        let visible = rows[top..end].to_vec();
        frame.render_widget(Paragraph::new(Text::from(visible)).block(block), area);
    }

    fn render_input(&self, frame: &mut Frame, area: Rect) {
        let block = Block::bordered().title(" command ");
        let para = Paragraph::new(self.input.as_str()).block(block);
        frame.render_widget(para, area);
        // Place the cursor after the typed text (inside the border).
        let cx = area.x + 1 + self.input.chars().count() as u16;
        frame.set_cursor_position((cx.min(area.x + area.width.saturating_sub(2)), area.y + 1));
    }
}

/// Flatten a styled line back to its plain text (headless output).
fn line_to_plain(line: &Line) -> String {
    line.spans.iter().map(|s| s.content.as_ref()).collect()
}

/// Hard-wrap one styled line into `width`-column rows, preserving span styles.
/// Character-based (not word-aware), which is fine for the structured sim output
/// and keeps row counts exact. An empty line yields one empty row.
fn wrap_line(line: &Line<'static>, width: usize) -> Vec<Line<'static>> {
    let width = width.max(1);
    let mut rows: Vec<Line<'static>> = Vec::new();
    let mut cur: Vec<Span<'static>> = Vec::new();
    let mut cur_w = 0usize;
    for span in &line.spans {
        let style = span.style;
        let mut buf = String::new();
        for ch in span.content.chars() {
            if cur_w == width {
                if !buf.is_empty() {
                    cur.push(Span::styled(std::mem::take(&mut buf), style));
                }
                rows.push(Line::from(std::mem::take(&mut cur)));
                cur_w = 0;
            }
            buf.push(ch);
            cur_w += 1;
        }
        if !buf.is_empty() {
            cur.push(Span::styled(buf, style));
        }
    }
    rows.push(Line::from(cur));
    rows
}
