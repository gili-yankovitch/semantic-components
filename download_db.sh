#!/usr/bin/env bash
set -euo pipefail

BASE_URL="https://yaqwsx.github.io/jlcparts/data"

for i in $(seq -w 8 18); do
  FILE="cache.z${i}"
  echo "Downloading ${FILE}..."
  wget "${BASE_URL}/${FILE}" &
done

wait

echo "All files downloaded."