#!/usr/bin/env bash

QUERY_NUM="q08"
FILENAME="${BIG_BENCH_QUERIES_DIR}/${QUERY_NUM}/${QUERY_NUM}.sql"
hive -f "$FILENAME"

echo "========================="
echo "Query results in: ${BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${QUERY_NUM}result"
echo "Display results : hadoop fs -cat ${BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${QUERY_NUM}result/*"
echo "========================="
