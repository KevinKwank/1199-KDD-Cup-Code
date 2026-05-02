#!/bin/bash
set -e

echo "=========================================="
echo "KDD Cup 2026 Data Agent"
echo "Model: ${MODEL_NAME:-not set}"
echo "=========================================="

mkdir -p /output /logs

python /app/main.py 2>&1 | tee /logs/runtime.log

echo "=========================================="
echo "Completed. Output files:"
find /output -name "prediction.csv" | wc -l
echo "=========================================="
