#!/usr/bin/env bash
"$BIG_BENCH_BASH_SCRIPT_DIR"/cleanBigBenchQueries.sh

echo "cleaning DROP TABLES"
hive -f "${BIG_BENCH_HIVE_SCRIPT_DIR}/dropTables.sql"
echo "cleaning ${BIG_BENCH_HDFS_ABSOLUTE_DATA_DIR}"
hadoop fs -rm -r -skipTrash "${BIG_BENCH_HDFS_ABSOLUTE_DATA_DIR}"
rm -rf "$BIG_BENCH_LOADING_STAGE_LOG"




