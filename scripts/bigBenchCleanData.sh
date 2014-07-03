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

#"$BIG_BENCH_BASH_SCRIPT_DIR/bigBenchCleanQueries.sh"

echo "cleaning DROP TABLES"
hive -f "${BIG_BENCH_HIVE_SCRIPT_DIR}/dropTables.sql"
echo "cleaning ${BIG_BENCH_HDFS_ABSOLUTE_DATA_DIR}"
hadoop fs -rm -r -skipTrash "${BIG_BENCH_HDFS_ABSOLUTE_DATA_DIR}"
rm -rf "$BIG_BENCH_LOADING_STAGE_LOG"
