#!/usr/bin/env bash
HIVE_PARAMS="--auxpath $BIG_BENCH_QUERIES_DIR/Resources/opennlp-maxent-3.0.3.jar:$BIG_BENCH_QUERIES_DIR/Resources/opennlp-tools-1.5.3.jar:${BIG_BENCH_QUERIES_DIR}/Resources/bigbenchqueriesmr.jar"
"${BIG_BENCH_HIVE_SCRIPT_DIR}/execQuery.sh" q19 "${HIVE_PARAMS}"
