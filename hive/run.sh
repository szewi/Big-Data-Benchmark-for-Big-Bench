#!/usr/bin/env bash
hive -f ${BIG_BENCH_HIVE_SCRIPT_DIR}/dropTables.hql
hive -f ${BIG_BENCH_HIVE_SCRIPT_DIR}/hiveCreateLoadOCR.hql

hadoop fs -mkdir "$BIG_BENCH_HDFS_ABSOLUTE_TEMP_DIR"
hadoop fs -mkdir "$BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR"
hadoop fs -chmod ugo+rw "$BIG_BENCH_HDFS_ABSOLUTE_TEMP_DIR"
hadoop fs -chmod ugo+rw "$BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR"
