#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p data

if [[ -f data/enwik8 ]]; then
    size=$(stat -c%s data/enwik8)
    if [[ "$size" -ge 100000000 ]]; then
        echo "data/enwik8 już istnieje (${size} B) — pomijam"
        exit 0
    fi
    echo "data/enwik8 istnieje, ale jest niepełny (${size} B) — pobieram ponownie"
    rm -f data/enwik8
fi

echo "[+] pobieram enwik8 (~36 MB skompresowane → 100 MB rozpakowane)..."
curl -L --fail --progress-bar -o data/enwik8.zip http://mattmahoney.net/dc/enwik8.zip
echo "[+] rozpakowuję..."
unzip -o -d data data/enwik8.zip
rm -f data/enwik8.zip

size=$(stat -c%s data/enwik8)
echo "[+] gotowe: data/enwik8 = ${size} B"

if [[ "$size" -ne 100000000 ]]; then
    echo "UWAGA: oczekiwano 100000000 B, jest ${size} B" >&2
    exit 1
fi
