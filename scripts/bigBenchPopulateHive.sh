#!/usr/bin/env bash

ENV_SETTINGS="`dirname $0`/../setEnvVars"
if [ ! -f "$ENV_SETTINGS" ]
then
        echo "Environment setup file $ENV_SETTINGS not found"
        exit 1
else
        source "$ENV_SETTINGS"
fi

echo "==============================================="
echo "Adding/Updating generated files to HIVE. (drops old tables)"
echo "==============================================="

time ("${BIG_BENCH_HIVE_SCRIPT_DIR}/hiveCreateLoadORC.sh" ; echo  "======= Load data into hive time =========") > >(tee -a "$BIG_BENCH_LOADING_STAGE_LOG") 2>&1 
echo "==========================="

echo "==============================================="
echo "HIVE load finished. You may start executing the queries by running script:"
echo " bigBenchRunQueries.sh"
echo "==============================================="
