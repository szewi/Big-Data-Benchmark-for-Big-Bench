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
	echo "==============================================="
	echo "Running query  : $QUERY_NAME"
	echo "-----------------------------------------------"
	echo "benchmark phase: $BENCHMARK_PHASE"
	echo "stream number  : $STREAM_NUMBER"
	echo "user parameter file : $USER_QUERY_PARAMS_FILE"
	echo "user settings file  :$USER_HIVE_SETTINGS_FILE"
	if [[ -n "$DEBUG_QUERY_PART" ]]; then
		echo "query part to debug: $DEBUG_QUERY_PART"
	fi
	echo "log: $LOG_FILE_NAME"
	echo "==============================================="	

	### Checking required folder: logs/; tmp/; result/ if they exist, create them and set permissions

	echo "checking existence of local: $BIG_BENCH_LOGS_DIR"
	if [ ! -d "$BIG_BENCH_LOGS_DIR" ]; then
		mkdir -p "$BIG_BENCH_LOGS_DIR"
	fi

	if [ ! -e "$LOG_FILE_NAME" ] ; then
    	touch "$LOG_FILE_NAME"
	fi

	if [ ! -w "$LOG_FILE_NAME" ] ; then
    	echo "ERROR: cannot write to: $LOG_FILE_NAME, no permission"
    	exit 1
	fi

	echo "creating folders and setting permissions"
	hadoop fs -rm -r -skipTrash "${RESULT_DIR}" &
	hadoop fs -rm -r -skipTrash "${TEMP_DIR}" &
	wait
	hadoop fs -mkdir -p "${RESULT_DIR}" &
	hadoop fs -mkdir -p "${TEMP_DIR}" &
	wait
	hadoop fs -chmod uga+rw "${BIG_BENCH_HDFS_ABSOLUTE_TEMP_DIR}" &
	hadoop fs -chmod uga+rw "${BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}" &
	hadoop fs -chmod uga+rw "${RESULT_DIR}" &
	hadoop fs -chmod uga+rw "${TEMP_DIR}" &
	wait

	# start timed execution of query. Stderr is appended to stdout and both are written into logs/q??.log and to console

	# Run the main method implemented in the query's run.sh
	time ("$QUERY_MAIN_METHOD" ; echo "======= $QUERY_NAME time =========") > >(tee -a "$LOG_FILE_NAME") 2>&1 
	echo "==========================="

	echo "======= $QUERY_NAME result =======" | tee -a "$LOG_FILE_NAME" 2>&1
	echo "results in: $RESULT_DIR" | tee -a "$LOG_FILE_NAME" 2>&1
	echo "to display: hadoop fs -ls $RESULT_DIR/*" | tee -a "$LOG_FILE_NAME" 2>&1
	echo "=========================" | tee -a "$LOG_FILE_NAME" 2>&1

	## append query specifc log to allInOne logfile
	#cat "$LOG_FILE_NAME" >> "$BIG_BENCH_LOGS_DIR/allQueries.log"
}

source "$SETUP_PARAMS"
