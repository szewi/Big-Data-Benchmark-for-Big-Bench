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

# check arguments
if [ "$1" = "-h" ]
then
	echo "usage: `basename "$0"` [-y <query parameter file>] [-z <hive settings file>] [-p <benchmark phase>] [-s <stream number>] [-d <query part to debug>]"
	exit 0
fi

FIRST_QUERY="1"
LAST_QUERY="30"

# parse command line arguments
while getopts ":y:z:p:s:d:" opt; do
	case $opt in
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

if [ -n "$USER_QUERY_PARAMS_FILE" ]
then
	if [ -r "$USER_QUERY_PARAMS_FILE" ]
	then
		RUN_QUERY_ARGS="-y $USER_QUERY_PARAMS_FILE"
	else
		echo "User query parameter file $USER_QUERY_PARAMS_FILE can not be read."
		exit 1
	fi
fi

if [ -n "$USER_HIVE_SETTINGS_FILE" ]
then
	if [ -r "$USER_HIVE_SETTINGS_FILE" ]
	then
		RUN_QUERY_ARGS="$RUN_QUERY_ARGS -z $USER_HIVE_SETTINGS_FILE"
	else
		echo "User hive settings file $USER_HIVE_SETTINGS_FILE can not be read."
		exit 1
	fi
fi

if [ -n "$USER_BENCHMARK_PHASE" ]
then
	RUN_QUERY_ARGS="$RUN_QUERY_ARGS -p $USER_BENCHMARK_PHASE"
fi

if [ -n "$USER_STREAM_NUMBER" ]
then
	RUN_QUERY_ARGS="$RUN_QUERY_ARGS -s $USER_STREAM_NUMBER"
fi

if [ -n "$DEBUG_QUERY_PART" ]
then
	RUN_QUERY_ARGS="$RUN_QUERY_ARGS -d $DEBUG_QUERY_PART"
fi

#echo "==============================================="
#echo "cleanup tmp files from previous query runs"
#echo "==============================================="
#"$BIG_BENCH_BASH_SCRIPT_DIR/bigBenchCleanQueries.sh"

echo "checking existence of: $BIG_BENCH_LOGS_DIR"
if [ ! -d "$BIG_BENCH_LOGS_DIR" ]; then
	if ! mkdir -p "$BIG_BENCH_LOGS_DIR" ; then
		echo "ERROR: cannot write to: $BIG_BENCH_LOGS_DIR, no permission"
		exit 1
	fi
fi

echo "==============================================="
echo "Running queries $FIRST_QUERY-$LAST_QUERY"
#echo "logging run to: $BIG_BENCH_LOGS_DIR/allQueries.log"
echo "==============================================="

for (( i=$FIRST_QUERY; i <=$LAST_QUERY; i++ ))
do
	"$BIG_BENCH_BASH_SCRIPT_DIR/bigBenchRunQuery.sh" -q $i $RUN_QUERY_ARGS
done

echo "==============================================="
echo "All queries finished"
echo "==============================================="
"$BIG_BENCH_BASH_SCRIPT_DIR/showErrors.sh" 

"$BIG_BENCH_BASH_SCRIPT_DIR/showTimes.sh" 
"$BIG_BENCH_BASH_SCRIPT_DIR/bigBenchZipQueryLogs.sh" 
