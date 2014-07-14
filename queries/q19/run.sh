#!/usr/bin/env bash

TEMP_TABLE1="${TEMP_TABLE}_sr_items"
TEMP_TABLE2="${TEMP_TABLE}_wr_items"
TEMP_TABLE3="${TEMP_TABLE}_return_items"

HIVE_PARAMS="$HIVE_PARAMS -hiveconf TEMP_TABLE1=$TEMP_TABLE1 -hiveconf TEMP_TABLE2=$TEMP_TABLE2 -hiveconf TEMP_TABLE3=$TEMP_TABLE3"

query_run_main_method () {
	HIVE_SCRIPT="$QUERY_DIR/$QUERY_NAME.sql"
	if [ ! -r "$HIVE_SCRIPT" ]
	then
		echo "SQL file $HIVE_SCRIPT can not be read."
		exit 1
	fi

	local HIVE_PARAMS="--auxpath $BIG_BENCH_QUERIES_DIR/Resources/opennlp-maxent-3.0.3.jar:$BIG_BENCH_QUERIES_DIR/Resources/opennlp-tools-1.5.3.jar:${BIG_BENCH_QUERIES_DIR}/Resources/bigbenchqueriesmr.jar $HIVE_PARAMS"

	runHiveCmd -f "$HIVE_SCRIPT"
}

query_run_clean_method () {
	runHiveCmd -e "DROP VIEW IF EXISTS $TEMP_TABLE1; DROP VIEW IF EXISTS $TEMP_TABLE2; DROP VIEW IF EXISTS $TEMP_TABLE3; DROP TABLE IF EXISTS $RESULT_TABLE;"
}
