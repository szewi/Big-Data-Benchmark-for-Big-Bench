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

# write environment information into logfile
logEnvInformation

GLOBAL_QUERY_PARAMS_FILE="$BIG_BENCH_HIVE_SCRIPT_DIR/queryParameters.sql"
GLOBAL_HIVE_SETTINGS_FILE="$BIG_BENCH_HIVE_SCRIPT_DIR/hiveSettings.sql"
GLOBAL_BENCHMARK_PHASE="RUN_QUERY"
GLOBAL_STREAM_NUMBER="0"

# check arguments
if [ $# -lt 2 ]
then
	echo "parameter missing"
	echo "usage: `basename "$0"` -q <query number> [-y <query parameter file>] [-z <hive settings file>] [-p <benchmark phase>] [-s <stream number>] [-d <query part to debug>]"
	exit 1
fi

# parse command line arguments
while getopts ":q:y:z:p:s:d:" opt; do
	case $opt in
		q)
			#echo "-q was triggered, Parameter: $OPTARG" >&2
			QUERY_NUMBER="$OPTARG"
		;;
		y)
			#echo "-y was triggered, Parameter: $OPTARG" >&2
			USER_QUERY_PARAMS_FILE="$OPTARG"
		;;
		z)
			#echo "-z was triggered, Parameter: $OPTARG" >&2
			USER_HIVE_SETTINGS_FILE="$OPTARG"
		;;
		p)
			#echo "-p was triggered, Parameter: $OPTARG" >&2
			USER_BENCHMARK_PHASE="$OPTARG"
		;;
		s)
			#echo "-s was triggered, Parameter: $OPTARG" >&2
			USER_STREAM_NUMBER="$OPTARG"
		;;
		d)
			#echo "-d was triggered, Parameter: $OPTARG" >&2
			DEBUG_QUERY_PART="$OPTARG"
		;;
		\?)
			echo "Invalid option: -$OPTARG" >&2
			exit 1
		;;
		:)
			echo "Option -$OPTARG requires an argument." >&2
			exit 1
		;;
	esac
done

# source the query's run.sh as early as possible
# this allows setting variables from run.sh which were not defined on command line
if [ -z "$QUERY_NUMBER" ]
then
	echo "The query number must be set."
	exit 1
fi

if [[ $QUERY_NUMBER -lt 1 || $QUERY_NUMBER -gt 30 ]]
then
	echo "Query number must be between 1 and 30"
	exit 1
fi

if [ $QUERY_NUMBER -lt 10 ]
then
	QUERY_NAME=q0$QUERY_NUMBER
else
	QUERY_NAME=q$QUERY_NUMBER
fi

QUERY_DIR="$BIG_BENCH_QUERIES_DIR/$QUERY_NAME"
if [ ! -d "$QUERY_DIR" ]
then
	echo "Query directory $QUERY_DIR does not exist"
	exit 1
fi

SCRIPT_FILENAME="$QUERY_DIR/run.sh"

if [ -r "$SCRIPT_FILENAME" ]
then
	source "$SCRIPT_FILENAME"
else
	echo "File $SCRIPT_FILENAME containing main method not found, aborting script."
	exit 1
fi

# check if the main method was implemented properly in the run.sh
MAIN_METHOD="query_run_main_method"
if ! declare -F "$MAIN_METHOD" > /dev/null 2>&1
then
	echo "main method was not implemented, aborting script"
	exit 1
fi

# check arguments and build all-in-one config file for hive
COMBINED_PARAMS_FILE="`mktemp`"
if [ -r "$GLOBAL_QUERY_PARAMS_FILE" ]
then
	cat "$GLOBAL_QUERY_PARAMS_FILE" > "$COMBINED_PARAMS_FILE"
else
	echo "Global query parameter file $GLOBAL_QUERY_PARAMS_FILE can not be read."
	rm -rf "$COMBINED_PARAMS_FILE"
	exit 1
fi

if [ -n "$USER_QUERY_PARAMS_FILE" ]
then
	if [ -r "$USER_QUERY_PARAMS_FILE" ]
	then
		cat "$USER_QUERY_PARAMS_FILE" >> "$COMBINED_PARAMS_FILE"
	else
		echo "User query parameter file $USER_QUERY_PARAMS_FILE can not be read."
		rm -rf "$COMBINED_PARAMS_FILE"
		exit 1
	fi
fi

LOCAL_HIVE_SETTINGS_FILE="$QUERY_DIR/hiveLocalSettings.sql"
if [ -r "$LOCAL_HIVE_SETTINGS_FILE" ]
then
	cat "$LOCAL_HIVE_SETTINGS_FILE" >> "$COMBINED_PARAMS_FILE"
fi

if [ -r "$GLOBAL_HIVE_SETTINGS_FILE" ]
then
	cat "$GLOBAL_HIVE_SETTINGS_FILE" >> "$COMBINED_PARAMS_FILE"
else
	echo "Global hive settings file $GLOBAL_HIVE_SETTINGS_FILE can not be read."
	rm -rf "$COMBINED_PARAMS_FILE"
	exit 1
fi

if [ -n "$USER_HIVE_SETTINGS_FILE" ]
then
	if [ -r "$USER_HIVE_SETTINGS_FILE" ]
	then
		cat "$GLOBAL_HIVE_SETTINGS_FILE" >> "$COMBINED_PARAMS_FILE"
	else
		echo "User hive settings file $USER_HIVE_SETTINGS_FILE can not be read."
		rm -rf "$COMBINED_PARAMS_FILE"
		exit 1
	fi
fi

BENCHMARK_PHASE="${USER_BENCHMARK_PHASE:-$GLOBAL_BENCHMARK_PHASE}"
STREAM_NUMBER="${USER_STREAM_NUMBER:-$GLOBAL_STREAM_NUMBER}"

RESULT_TABLE="${QUERY_NAME}_${BENCHMARK_PHASE}_${STREAM_NUMBER}_result"
RESULT_DIR="$BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR/$RESULT_TABLE"
TEMP_TABLE="${QUERY_NAME}_${BENCHMARK_PHASE}_${STREAM_NUMBER}_temp"
TEMP_DIR="$BIG_BENCH_HDFS_ABSOLUTE_TEMP_DIR/$TEMP_TABLE"

LOG_FILE_NAME="$BIG_BENCH_LOGS_DIR/${QUERY_NAME}_${BENCHMARK_PHASE}_${STREAM_NUMBER}.log"

HIVE_PARAMS="-hiveconf BENCHMARK_PHASE=$BENCHMARK_PHASE -hiveconf STREAM_NUMBER=$STREAM_NUMBER -hiveconf QUERY_NAME=$QUERY_NAME -hiveconf QUERY_DIR=$QUERY_DIR -hiveconf RESULT_TABLE=$RESULT_TABLE -hiveconf RESULT_DIR=$RESULT_DIR -hiveconf TEMP_TABLE=$TEMP_TABLE -hiveconf TEMP_DIR=$TEMP_DIR"

echo "==============================================="
echo "Running query: $QUERY_NAME"
echo "log: $LOG_FILE_NAME"
echo "==============================================="	

### Checking required folder: logs/; tmp/; result/ if they exist, create them and set permissions 

echo "checking existence of: $BIG_BENCH_LOGS_DIR"
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
time ("$MAIN_METHOD" ; echo "======= $QUERY_NAME time =========") > >(tee -a "$LOG_FILE_NAME") 2>&1 
echo "==========================="

echo "======= $QUERY_NAME result =======" | tee -a "$LOG_FILE_NAME" 2>&1
echo "results in: $RESULT_DIR" | tee -a "$LOG_FILE_NAME" 2>&1
echo "to display: hadoop fs -ls $RESULT_DIR/*" | tee -a "$LOG_FILE_NAME" 2>&1
echo "=========================" | tee -a "$LOG_FILE_NAME" 2>&1

## append query specifc log to allInOne logfile
#cat "$LOG_FILE_NAME" >> "$BIG_BENCH_LOGS_DIR/allQueries.log"

rm -rf "$COMBINED_PARAMS_FILE"
