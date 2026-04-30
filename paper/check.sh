#!/usr/bin/env bash
# paper/check.sh — LaTeX linter + clean-build sanity check.
#
# Runs three independent checks and surfaces only the issues that
# matter (errors, undefined refs, overfull boxes, hyperref unicode
# warnings, ChkTeX style problems). Cosmetic noise (Underfull \vbox,
# float specifier rewrites, "Label may have changed" first-pass
# warnings) is filtered out.
#
# Usage:
#     cd paper && ./check.sh                    # checks main.tex (arXiv preprint)
#     cd paper && ./check.sh main_tmlr.tex      # checks TMLR anon submission
#     cd paper && ./check.sh --strict           # fail on style issues too
#
# Exit codes: 0 = clean, 1 = real issues.

set -eo pipefail
cd "$(dirname "$0")"

CHKTEX_BIN=$(command -v chktex || echo /tmp/chktex_extracted/usr/bin/chktex)
CHKTEX_RC=/tmp/chktex_extracted/etc/chktexrc
STRICT=0
ENTRY="main.tex"
for arg in "$@"; do
    case "$arg" in
        --strict) STRICT=1 ;;
        *.tex)    ENTRY="$arg" ;;
    esac
done
JOB="${ENTRY%.tex}"
echo "==> Target: $ENTRY (jobname $JOB)"

# ---------- 1. Clean build ----------
echo "==> Clean rebuild (pdflatex + bibtex + pdflatex x2)"
rm -f $JOB.aux $JOB.bbl $JOB.blg $JOB.log $JOB.out $JOB.toc 2>/dev/null
pdflatex -interaction=nonstopmode -draftmode $ENTRY >/dev/null 2>&1 || true
bibtex $JOB >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode -draftmode $ENTRY >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode $ENTRY >$JOB.check.log 2>&1 || true

real_errors=0

# ---------- 2. Errors and undefined refs ----------
echo "==> Compile errors"
if grep -E "^!" $JOB.check.log >/dev/null 2>&1; then
    grep -E "^!" $JOB.check.log
    real_errors=$((real_errors + 1))
else
    echo "    none"
fi

echo "==> Undefined citations / references"
undef=$(grep -cE "Citation .* undefined|Reference .* undefined" $JOB.check.log || true)
if [[ "$undef" != "0" ]]; then
    grep -E "Citation .* undefined|Reference .* undefined" $JOB.check.log | sort -u | head -10
    real_errors=$((real_errors + 1))
else
    echo "    none"
fi

# ---------- 3. Overfull hboxes (table/figure overflow) ----------
echo "==> Overfull hboxes (>= 5pt)"
overfull=$(grep -E "Overfull \\\\hbox \([0-9.]+pt too wide\)" $JOB.check.log \
    | awk -F'[() ]' '$3 > 5 {print}' || true)
if [[ -n "$overfull" ]]; then
    echo "$overfull" | head -10
    real_errors=$((real_errors + 1))
else
    echo "    none"
fi

# ---------- 4. Hyperref unicode warnings (broken PDF bookmarks) ----------
echo "==> hyperref Token-not-allowed (math in section titles)"
hyperref=$(grep -c "Token not allowed in a PDF string" $JOB.check.log || true)
if [[ "$hyperref" -gt 0 ]]; then
    echo "    $hyperref warnings — wrap math in \\texorpdfstring{...}{...}"
    grep -B1 "Token not allowed" $JOB.check.log | grep "input line" | sort -u | head -5
    real_errors=$((real_errors + 1))
else
    echo "    none"
fi

# ---------- 5. Float specifier rewrites ----------
echo "==> Float-specifier rewrites ([h] -> [ht])"
floats=$(grep -c "float specifier changed" $JOB.check.log || true)
if [[ "$floats" -gt 0 ]]; then
    echo "    $floats — replace [h] with [!t] or [!htbp]"
    real_errors=$((real_errors + 1))
else
    echo "    none"
fi

# ---------- 6. ChkTeX style pass ----------
if [[ -x "$CHKTEX_BIN" ]]; then
    echo "==> ChkTeX style pass (-q quiet, suppress 8/22/40 = false-positive heavy)"
    # -n8: suppress wrong-spacing-after-period (heavy false positives)
    # -n22: suppress comments-in-front-of-tex-source
    # -n40: suppress raised punctuation
    chktex_out=$("$CHKTEX_BIN" -q -n8 -n22 -n40 \
        ${CHKTEX_RC:+-l "$CHKTEX_RC"} \
        $ENTRY sections/*.tex appendix/*.tex 2>/dev/null \
        | grep -E "^(Warning|Error|Message)" || true)
    if [[ -n "$chktex_out" ]]; then
        echo "$chktex_out" | head -20
        echo "    ($(echo "$chktex_out" | wc -l) total ChkTeX issues)"
        if [[ "$STRICT" == "1" ]]; then
            real_errors=$((real_errors + 1))
        fi
    else
        echo "    none"
    fi
else
    echo "==> ChkTeX not available — skipping style pass"
fi

# ---------- 7. Pages and size ----------
echo
echo "==> Output"
grep -E "Output written" $JOB.check.log || echo "    no PDF produced"

# ---------- Result ----------
echo
if [[ "$real_errors" -eq 0 ]]; then
    echo "Result: CLEAN."
    rm -f $JOB.check.log
    exit 0
else
    echo "Result: $real_errors issue group(s) — see details above; full log in paper/$JOB.check.log"
    exit 1
fi
