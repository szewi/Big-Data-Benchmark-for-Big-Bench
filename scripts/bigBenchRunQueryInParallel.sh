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
if [[ $# -eq 0 || "$1" = "-h" ]]
then
	echo "usage: `basename "$0"` -q <query number> [-y <query parameter file>] [-z <hive settings file>] [-p <benchmark phase>] -s <number of parallel streams> [-d <query part to debug>]"
	exit 0
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

if [ -n "$QUERY_NUMBER" ]
then
	RUN_QUERY_ARGS="-q $QUERY_NUMBER"
else
	echo "Query number option is required"
	exit 1
fi

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
	NUMBER_OF_PARALLEL_STREAMS="$USER_STREAM_NUMBER"
else
	echo "The number of parallel streams option is required"
	exit 1
fi

if [ -n "$DEBUG_QUERY_PART" ]
then
	RUN_QUERY_ARGS="$RUN_QUERY_ARGS -d $DEBUG_QUERY_PART"
fi

for (( i = 0; i < $NUMBER_OF_PARALLEL_STREAMS; i++ ))
do
	"$BIG_BENCH_BASH_SCRIPT_DIR/bigBenchRunQuery.sh" -s $i $RUN_QUERY_ARGS &
done
wait
