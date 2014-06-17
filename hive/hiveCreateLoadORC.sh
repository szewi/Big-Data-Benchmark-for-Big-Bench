#!/usr/bin/env bash

ENV_SETTINGS="`dirname $0`/../setEnvVars"
if [ ! -f "$ENV_SETTINGS" ]
then
        echo "Environment setup file $ENV_SETTINGS not found"
        exit 1
else
        source "$ENV_SETTINGS"
fi

hive -f ${BIG_BENCH_HIVE_SCRIPT_DIR}/dropTables.sql
hive -f ${BIG_BENCH_HIVE_SCRIPT_DIR}/hiveCreateLoadORC.sql

hadoop fs -mkdir -p "$BIG_BENCH_HDFS_ABSOLUTE_TEMP_DIR"
hadoop fs -mkdir -p "$BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR"
hadoop fs -chmod ugo+rw "$BIG_BENCH_HDFS_ABSOLUTE_TEMP_DIR"
hadoop fs -chmod ugo+rw "$BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR"
