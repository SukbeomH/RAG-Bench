#!/usr/bin/env bash
# 벤치마크 PDF 11개를 worker_structured로 일괄 처리
set -euo pipefail

PADDLEOCR_PYTHON="/Users/sukbeom/Desktop/autorag/PaddleOCR/.venv/bin/python"
WORKER="/Users/sukbeom/Desktop/autorag/isolated_backends/paddleocr/worker_structured.py"
OUTDIR="/Users/sukbeom/Desktop/autorag/data/benchmark_pdfs/paddleocr_raw"
mkdir -p "$OUTDIR"

unset VIRTUAL_ENV

for pdf in /Users/sukbeom/Desktop/autorag/data/benchmark_pdfs/*.pdf; do
    name=$(basename "$pdf" .pdf)
    outfile="$OUTDIR/${name}.json"

    if [ -f "$outfile" ]; then
        echo "SKIP (exists): $name"
        continue
    fi

    echo "Processing: $name ..."
    if "$PADDLEOCR_PYTHON" "$WORKER" "$pdf" > "/tmp/bench_${name}.txt" 2>/tmp/bench_${name}_err.txt; then
        # Extract JSON between tokens
        python3 -c "
import sys
txt = open('/tmp/bench_${name}.txt').read()
s = txt.find('---OUTPUT_START---')
e = txt.find('---OUTPUT_END---')
if s >= 0 and e > s:
    print(txt[s+len('---OUTPUT_START---'):e].strip())
else:
    print('ERROR: tokens not found', file=sys.stderr)
    sys.exit(1)
" > "$outfile"
        echo "  OK → $outfile"
    else
        echo "  FAILED: $name"
        cat "/tmp/bench_${name}_err.txt" | tail -5
    fi
done

echo ""
echo "Done. Files:"
ls -lh "$OUTDIR"/*.json 2>/dev/null || echo "No output files"
