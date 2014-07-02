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

run_main_method () {
	echo "drop old result table in $RESULT_DIR"
	hive $HIVE_PARAMS -i "$COMBINED_PARAMS_FILE" -f "$BIG_BENCH_QUERIES_DIR/dropResult.sql"

	echo "cleaning dir $RESULT_DIR"
	hadoop fs -rm -r -skipTrash "$RESULT_DIR" &

	echo "cleaning dir $TEMP_DIR"
	hadoop fs -rm -r -skipTrash "$TEMP_DIR" &

	wait

	echo "cleaning log $LOG_FILE_NAME"
	rm -rf "$LOG_FILE_NAME"
}

source "$SETUP_PARAMS"
