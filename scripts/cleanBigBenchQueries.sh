#!/usr/bin/env bash

echo "drop old result tabls in ${BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}"
hive -f $BIG_BENCH_QUERIES_DIR/dropAllResults.sql

echo "cleaning dir ${BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}"
hadoop fs -rm -r -skipTrash "${BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}"

echo "cleaning dir ${BIG_BENCH_HDFS_ABSOLUTE_TEMP_DIR}"
hadoop fs -rm -r -skipTrash "${BIG_BENCH_HDFS_ABSOLUTE_TEMP_DIR}"

echo "cleaning dir $BIG_BENCH_LOGS_DIR"
rm -rf "$BIG_BENCH_LOGS_DIR"/q??.log
rm -rf "$BIG_BENCH_LOGS_DIR"/allQueries.log

