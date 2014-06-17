#!/usr/bin/env bash

ENV_SETTINGS="`dirname $0`/../../setEnvVars"
if [ ! -f "$ENV_SETTINGS" ]
then
        echo "Environment setup file $ENV_SETTINGS not found"
        exit 1
else
        source "$ENV_SETTINGS"
fi

QUERY_NUM="q23"
QUERY_DIR="${BIG_BENCH_QUERIES_DIR}/${QUERY_NUM}"

echo "========================="
echo "$QUERY_NUM Step 1/4: make view"
echo "========================="
hive   -f "${QUERY_DIR}/q23_1.sql"

echo "========================="
echo "$QUERY_NUM Step 2/4: make result 1"
echo "========================="
hive   -f "${QUERY_DIR}/q23_2.sql"

echo "========================="
echo "$QUERY_NUM Step 3/4: make result 2"
echo "========================="
hive   -f "${QUERY_DIR}/q23_3.sql"


echo "========================="
echo "$QUERY_NUM Step 4/4: cleanup"
echo "========================="
hive   -f "${QUERY_DIR}/cleanup.sql"


echo "======= $QUERY_NUM  result ======="
echo "result1 in: ${BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${QUERY_NUM}result1"
echo "to display : hadoop fs -cat ${BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${QUERY_NUM}result1/*"
echo "result2 in: ${BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${QUERY_NUM}result2"
echo "to display : hadoop fs -cat ${BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${QUERY_NUM}result2/*"
echo "========================="


