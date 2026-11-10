#!/bin/bash
# Probe Kaggle every 2 minutes until the main competition's rules are accepted,
# then run the real download. Gives up after 12 hours.
cd "$(dirname "$0")/.." || exit 1
mkdir -p data/kaggle/_probe
for i in $(seq 1 360); do
  rm -f data/kaggle/_probe/*
  .venv/bin/kaggle competitions download -c gemma-4-developer-agent -f HARNESS_README.md -p data/kaggle/_probe -q >/dev/null 2>&1
  if [ -n "$(ls -A data/kaggle/_probe 2>/dev/null)" ]; then
    echo "access granted after $i probes"
    .venv/bin/python src/download.py
    exit $?
  fi
  sleep 120
done
echo "gave up: rules still not accepted after 12h"
exit 2
