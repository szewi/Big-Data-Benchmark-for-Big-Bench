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

SETUP_PARAMS="`dirname $0`/bigBenchSetupQueryParams.sh"
if [ ! -f "$SETUP_PARAMS" ]
then
	echo "Query parameter setup file $SETUP_PARAMS not found"
	exit 1
fi

# write environment information into logfile
logEnvInformation

populateHive () {
	"$BIG_BENCH_BASH_SCRIPT_DIR/bigBenchCleanHive.sh"
	"$BINARY" $HIVE_PARAMS -i "$COMBINED_PARAMS_FILE" -f "${BIG_BENCH_HIVE_SCRIPT_DIR}/hiveCreateLoadORC.sql"

	hadoop fs -mkdir -p "$BIG_BENCH_HDFS_ABSOLUTE_TEMP_DIR" &
	hadoop fs -mkdir -p "$BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR" &
	wait
	hadoop fs -chmod ugo+rw "$BIG_BENCH_HDFS_ABSOLUTE_TEMP_DIR" &
	hadoop fs -chmod ugo+rw "$BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR" &
	wait
}

run_main_method () {
	echo "==============================================="
	echo "Adding/Updating generated files to HIVE. (drops old tables)"
	echo "==============================================="

	time (populateHive ; echo "======= Load data into hive time =========") > >(tee -a "$BIG_BENCH_LOADING_STAGE_LOG") 2>&1 
	echo "==========================="

	echo "==============================================="
	echo "HIVE load finished. You may start executing the queries by running script:"
	echo " bigBenchRunQueries.sh"
	echo "==============================================="
}

source "$SETUP_PARAMS"
