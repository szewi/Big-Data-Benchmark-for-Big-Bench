#!/usr/bin/env bash

# Source basic environment
ENV_SETTINGS="`dirname $0`/../setEnvVars"
if [ ! -f "$ENV_SETTINGS" ]
then
	echo "Environment setup file $ENV_SETTINGS not found"
	exit 1
else
	source "$ENV_SETTINGS"
fi

echo "drop old log $BIG_BENCH_LOADING_STAGE_LOG"
rm -rf "$BIG_BENCH_LOADING_STAGE_LOG"

# write environment information into logfile
logEnvInformation

populateHive () {
	"$BIG_BENCH_BASH_SCRIPT_DIR/bigBenchCleanHive.sh"
	"$HIVE_BINARY" -f "${BIG_BENCH_HIVE_SCRIPT_DIR}/hiveCreateLoadORC.sql"

	hadoop fs -mkdir -p "$BIG_BENCH_HDFS_ABSOLUTE_TEMP_DIR" &
	hadoop fs -mkdir -p "$BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR" &
	wait
	hadoop fs -chmod ugo+rw "$BIG_BENCH_HDFS_ABSOLUTE_TEMP_DIR" &
	hadoop fs -chmod ugo+rw "$BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR" &
	wait
}

echo "==============================================="
echo "Adding/Updating generated files to HIVE. (drops old tables)"
echo "==============================================="



time (populateHive ; echo "======= Load data into hive time =========") > >(tee -a "$BIG_BENCH_LOADING_STAGE_LOG") 2>&1 
echo "==========================="

echo "==============================================="
echo "HIVE load finished. You may start executing the queries by running script:"
echo " bigBenchRunQueries.sh"
echo "==============================================="
