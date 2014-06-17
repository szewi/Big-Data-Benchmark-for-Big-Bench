#!/usr/bin/env bash

ENV_SETTINGS="`dirname $0`/../setEnvVars"
if [ ! -f "$ENV_SETTINGS" ]
then
        echo "Environment setup file $ENV_SETTINGS not found"
        exit 1
else
        source "$ENV_SETTINGS"
fi

logEnvInformation

if [ $# -lt 1 ]
then
	echo "parameter missing"
	echo "usage: <queryNumber e.g.: q01>  <(optional)hive params e.g.: --auxpath>"
	exit 1
fi

QUERY_NUM="$1"
FILENAME="${BIG_BENCH_QUERIES_DIR}/${QUERY_NUM}/${QUERY_NUM}.sql"

if [ -f "$FILENAME" ]
then
	hive $2 -f "$FILENAME"
	echo "======= $QUERY_NUM  result ======="
	echo "results in : ${BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${QUERY_NUM}result"
	echo "to display : hadoop fs -cat ${BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${QUERY_NUM}result/*"
	echo "========================="
else
	echo "$FILENAME does not exist"
fi
