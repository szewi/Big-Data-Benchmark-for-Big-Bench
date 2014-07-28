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

echo "cleaning ${BIG_BENCH_HDFS_ABSOLUTE_INIT_DATA_DIR}"
hadoop fs -rm -r -skipTrash "${BIG_BENCH_HDFS_ABSOLUTE_INIT_DATA_DIR}"
