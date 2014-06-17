#!/usr/bin/env bash

ENV_SETTINGS="`dirname $0`/../../setEnvVars"
if [ ! -f "$ENV_SETTINGS" ]
then
        echo "Environment setup file $ENV_SETTINGS not found"
        exit 1
else
        source "$ENV_SETTINGS"
fi

QUERY_NUM="q08"
FILENAME="${BIG_BENCH_QUERIES_DIR}/${QUERY_NUM}/${QUERY_NUM}.sql"
hive -f "$FILENAME"

echo "========================="
echo "Query results in: ${BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${QUERY_NUM}result"
echo "Display results : hadoop fs -cat ${BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${QUERY_NUM}result/*"
echo "========================="
