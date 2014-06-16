#!/usr/bin/env bash
hive -f ${BIG_BENCH_HIVE_SCRIPT_DIR}/dropTables.sql
hive -f ${BIG_BENCH_HIVE_SCRIPT_DIR}/hiveCreateLoadORC.sql

hadoop fs -mkdir -p "$BIG_BENCH_HDFS_ABSOLUTE_TEMP_DIR"
hadoop fs -mkdir -p "$BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR"
hadoop fs -chmod ugo+rw "$BIG_BENCH_HDFS_ABSOLUTE_TEMP_DIR"
hadoop fs -chmod ugo+rw "$BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR"
