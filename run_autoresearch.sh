#!/bin/bash
# ==============================================================================
# run_autoresearch.sh -- Codex autoresearch loop for NeuroGenesis
# ==============================================================================
#
# Bash owns the outer loop. Each `codex exec` call is stateless and runs exactly
# one experiment cycle, then exits. Persistent state lives in git history plus
# `.autoresearch_archive/results.tsv`.
#
# Usage:
#   ./run_autoresearch.sh <model> <tag> [max_hours]
#
# Example:
#   ./run_autoresearch.sh gpt-5.4 mar26b 6
# ==============================================================================

set -euo pipefail

MODEL="${1:?Usage: $0 <model> <tag> [max_hours]}"
TAG="${2:?Usage: $0 <model> <tag> [max_hours]}"
MAX_HOURS="${3:-6}"
MAX_SECONDS=$((MAX_HOURS * 3600))

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE_REPO="${AUTORESEARCH_REPO:-$SCRIPT_DIR}"
WORK_DIR="${AUTORESEARCH_WORKDIR:-$HOME/neurogenesis_autoresearch_${TAG}}"
BRANCH="autoresearch/${TAG}"

ARCHIVE_DIR_REL=".autoresearch_archive"
RESULTS_REL="${ARCHIVE_DIR_REL}/results.tsv"
RUN_LOG_REL="${ARCHIVE_DIR_REL}/run.log"
TIMING_REL="${ARCHIVE_DIR_REL}/timing.log"
OUTPUT_REL="${ARCHIVE_DIR_REL}/output.log"
ARTIFACTS_REL="${ARCHIVE_DIR_REL}/artifacts/validation"

ARCHIVE_DIR="${WORK_DIR}/${ARCHIVE_DIR_REL}"
RESULTS_FILE="${WORK_DIR}/${RESULTS_REL}"
RUN_LOG="${WORK_DIR}/${RUN_LOG_REL}"
TIMING_LOG="${WORK_DIR}/${TIMING_REL}"
OUTPUT_LOG="${WORK_DIR}/${OUTPUT_REL}"
ARTIFACTS_DIR="${WORK_DIR}/${ARTIFACTS_REL}"

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY is not set."
    echo "Export it before running, for example:"
    echo "  export OPENAI_API_KEY='sk-...'"
    exit 1
fi

if [ ! -d "$SOURCE_REPO/.git" ]; then
    echo "ERROR: Cannot find a git repo at $SOURCE_REPO"
    echo "Set AUTORESEARCH_REPO=/path/to/NeuroGenesis if needed."
    exit 1
fi

BASE_COMMIT="${BASE_COMMIT:-$(git -C "$SOURCE_REPO" rev-parse HEAD)}"
echo "Base commit: $BASE_COMMIT"

if [ ! -d "$WORK_DIR/.git" ]; then
    echo "Creating local clone at $WORK_DIR ..."
    git clone --shared "$SOURCE_REPO" "$WORK_DIR"
fi

cd "$WORK_DIR"

git checkout master 2>/dev/null || git checkout main 2>/dev/null || true
git reset --hard "$BASE_COMMIT"
echo "Reset working copy to $BASE_COMMIT"

git branch -D "$BRANCH" 2>/dev/null || true
git checkout -b "$BRANCH"
echo "Created fresh branch $BRANCH from $BASE_COMMIT"

mkdir -p "$ARTIFACTS_DIR"
rm -f "$RESULTS_FILE" "$TIMING_LOG" "$OUTPUT_LOG" "$RUN_LOG"
rm -rf "$ARTIFACTS_DIR"
mkdir -p "$ARTIFACTS_DIR"
printf 'commit\tscore\ttime_s\tstatus\tdescription\n' > "$RESULTS_FILE"

echo "========================================" | tee -a "$TIMING_LOG"
echo "Autoresearch Run - $(date -Iseconds)"   | tee -a "$TIMING_LOG"
echo "Project:   NeuroGenesis"                | tee -a "$TIMING_LOG"
echo "Model:     $MODEL"                      | tee -a "$TIMING_LOG"
echo "Tag:       $TAG"                        | tee -a "$TIMING_LOG"
echo "Branch:    $BRANCH"                     | tee -a "$TIMING_LOG"
echo "Source:    $SOURCE_REPO"                | tee -a "$TIMING_LOG"
echo "WorkDir:   $WORK_DIR"                   | tee -a "$TIMING_LOG"
echo "Archive:   $ARCHIVE_DIR"                | tee -a "$TIMING_LOG"
echo "Max hours: $MAX_HOURS"                  | tee -a "$TIMING_LOG"
echo "========================================" | tee -a "$TIMING_LOG"

RUN_START=$(date +%s)
ITER=0

while true; do
    NOW=$(date +%s)
    ELAPSED_TOTAL=$((NOW - RUN_START))
    if [ "$ELAPSED_TOTAL" -ge "$MAX_SECONDS" ]; then
        echo "[${TAG}] Time limit reached (${MAX_HOURS}h). Stopping after ${ITER} iterations." | tee -a "$TIMING_LOG"
        break
    fi

    ITER=$((ITER + 1))
    ITER_START=$(date +%s)
    REMAINING_MIN=$(( (MAX_SECONDS - ELAPSED_TOTAL) / 60 ))
    ITER_ARTIFACT_REL="${ARTIFACTS_REL}/iter_${ITER}"

    echo "[${TAG}] iter=${ITER} start=$(date -Iseconds) remaining=${REMAINING_MIN}min model=${MODEL}" | tee -a "$TIMING_LOG"

    PROMPT="You are running one autonomous NeuroGenesis experiment cycle.

Read AUTORESEARCH.md and AGENTS.md fully before acting, then follow the repo's experiment loop.

State model:
- This codex invocation is stateless and exits after one experiment cycle.
- Persistent state lives only in git history and ${RESULTS_REL}.
- At the start of this iteration, you must inspect git state and read ${RESULTS_REL} before choosing what to do.

Archive rules:
- Put all autoresearch logs and outputs under ${ARCHIVE_DIR_REL}/.
- Use ${RUN_LOG_REL} for the validation log.
- Use ${RESULTS_REL} for the experiment ledger.
- Use ${ITER_ARTIFACT_REL} as the validation harness output directory.
- Do not leave run logs, TSVs, or experiment artifacts in the repo root.

Project-specific rules:
- This repo's validation harness already averages multiple seeds by default.
- Run: cargo run -p sim-validation --release -- --ticks 50000 --out ${ITER_ARTIFACT_REL} > ${RUN_LOG_REL} 2>&1
- After the run, use: grep \"^aggregate_score:\\|^aggregate_score_median:\\|^aggregate_score_stddev:\\|^aggregate_score_min:\\|^aggregate_score_max:\\|^total_time_seconds:\\|^seed_score\\|^seeds:\" ${RUN_LOG_REL}
- If grep output is empty, the run crashed. Inspect with: tail -n 50 ${RUN_LOG_REL}
- Log every result to ${RESULTS_REL} as tab-separated columns:
  commit<TAB>score<TAB>time_s<TAB>status<TAB>description
- Do not git-commit ${RESULTS_REL}
- If a run crashes and you cannot fix it quickly, log a crash row with score 0.00 and time 0.0
- Prefer changes that improve aggregate_score without an obvious median/min regression.
- If aggregate_score is equal or worse, discard it with: git reset --hard HEAD~1
- On iteration 1, establish the baseline by running the harness unmodified.

Search preferences:
- Prefer ambitious mechanism changes in sim-core over trivial config nudges.
- Do not add new speculative tests unless required to keep the existing test suite passing.
- Respect the allowed/disallowed file boundaries in AUTORESEARCH.md.

This is iteration ${ITER}. Run exactly ONE complete experiment cycle:
1. Read ${RESULTS_REL} and determine the current best score and recent failed ideas.
2. Read git branch/commit/state so you know exactly what code you are starting from.
3. If no baseline exists yet, run the unmodified baseline.
4. Otherwise, propose one change in the allowed sim-core/config files, implement it, and git commit it.
5. Run cargo test --workspace.
6. Run the validation harness in release mode, writing logs and artifacts under ${ARCHIVE_DIR_REL}/.
7. Read the results, append one row to ${RESULTS_REL}, and keep or discard the commit.
Then stop. The shell wrapper will call you again for the next iteration."

    codex exec \
        -m "$MODEL" \
        --dangerously-bypass-approvals-and-sandbox \
        "$PROMPT" \
        2>&1 | tee -a "$OUTPUT_LOG" || true

    ITER_END=$(date +%s)
    ITER_ELAPSED=$((ITER_END - ITER_START))

    LATEST_SCORE="N/A"
    TOTAL_ROWS=0
    if [ -f "$RESULTS_FILE" ]; then
        TOTAL_ROWS=$(( $(wc -l < "$RESULTS_FILE") - 1 ))
        if [ "$TOTAL_ROWS" -gt 0 ]; then
            LATEST_SCORE=$(tail -1 "$RESULTS_FILE" | cut -f2)
        fi
    fi

    echo "[${TAG}] iter=${ITER} elapsed=${ITER_ELAPSED}s experiments=${TOTAL_ROWS} latest_score=${LATEST_SCORE}" | tee -a "$TIMING_LOG"
    echo "---" >> "$TIMING_LOG"
done

echo "" | tee -a "$TIMING_LOG"
echo "========== FINAL RESULTS ==========" | tee -a "$TIMING_LOG"
echo "Project: NeuroGenesis"             | tee -a "$TIMING_LOG"
echo "Model:   $MODEL"                   | tee -a "$TIMING_LOG"
echo "Branch:  $BRANCH"                  | tee -a "$TIMING_LOG"
echo "Total iterations: $ITER"           | tee -a "$TIMING_LOG"
echo "Total time: $(($(date +%s) - RUN_START))s" | tee -a "$TIMING_LOG"
echo "Archive: ${ARCHIVE_DIR}"           | tee -a "$TIMING_LOG"
if [ -f "$RESULTS_FILE" ]; then
    echo "" | tee -a "$TIMING_LOG"
    echo "Results:" | tee -a "$TIMING_LOG"
    cat "$RESULTS_FILE" | tee -a "$TIMING_LOG"
fi
echo "====================================" | tee -a "$TIMING_LOG"
