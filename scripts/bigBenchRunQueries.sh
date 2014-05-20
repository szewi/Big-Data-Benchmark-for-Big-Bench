#!/usr/bin/env bash


FIRST_QUERY=1
if [ $# -gt 0 ]; then
	if [ $1 -gt 0 ]; then
		if [ $1 -lt 31 ]; then
			FIRST_QUERY=$1
		else
			echo "first argument must be in range [1,30]. was: $1" 
			exit 1
		fi
	else
		echo "first argument must be in range [1,30]. was: $1"
		exit 1
	fi
fi

LAST_QUERY=30

if [ $# -gt 1 ]; then
	if [ $2 -gt 0 ]; then
		if [ $2 -lt 31 ]; then
			if [ $1 -gt $2 ]; then
				echo "first argument \"$1\" must be lower or equal second argumet \"$2\""
				exit 1
			fi
		        LAST_QUERY=$2

		else
		        echo "second argument must be in range [1,30]. was: $2"
			exit 1
		fi
	else
		echo "second argument must be in range [1,30]. was: $2"
		exit 1
	fi
fi





echo "==============================================="
echo "cleanup tmp files from previous query runs"
echo "==============================================="
"$BIG_BENCH_BASH_SCRIPT_DIR"/cleanBigBenchQueries.sh



echo "checking existance of: $BIG_BENCH_LOGS_DIR "
if [ ! -d "$BIG_BENCH_LOGS_DIR" ]; then
	if ! mkdir -p "$BIG_BENCH_LOGS_DIR" ; then
		echo "ERROR: cannot write to: $BIG_BENCH_LOGS_DIR, no permission"
		exit 1
	fi
	
fi




echo "==============================================="
echo "Running queries $FIRST_QUERY-$LAST_QUERY"
echo "logging run to: $BIG_BENCH_LOGS_DIR/allQueries.log"
echo "==============================================="



for (( i=$FIRST_QUERY; i <=$LAST_QUERY; i++ ))
do

	"$BIG_BENCH_BASH_SCRIPT_DIR/bigBenchRunQuery.sh" ${i}

done

echo "==============================================="
echo "All queries finished"
echo "A log of this run is in: $BIG_BENCH_LOGS_DIR/allQueries.log"
echo "==============================================="
"$BIG_BENCH_BASH_SCRIPT_DIR/showErrors.sh" 

"$BIG_BENCH_BASH_SCRIPT_DIR/showTimes.sh" 


