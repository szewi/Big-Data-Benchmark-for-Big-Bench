#!/usr/bin/env bash

query_run_main_method () {
	HIVE_SCRIPT="$QUERY_DIR/$QUERY_NAME.sql"
	if [ ! -r "$HIVE_SCRIPT" ]
	then
		echo "SQL file $HIVE_SCRIPT can not be read."
		exit 1
	fi

	TEMP_TABLE1="${TABLE_PREFIX}_tmp_sr_items"
	TEMP_TABLE2="${TABLE_PREFIX}_tmp_wr_items"
	TEMP_TABLE3="${TABLE_PREFIX}_tmp_return_items"

	HIVE_PARAMS="$HIVE_PARAMS -hiveconf TEMP_TABLE1=$TEMP_TABLE1 -hiveconf TEMP_TABLE2=$TEMP_TABLE2 -hiveconf TEMP_TABLE3=$TEMP_TABLE3"

	HIVE_AUXPATH="--auxpath $BIG_BENCH_QUERIES_DIR/Resources/opennlp-maxent-3.0.3.jar:$BIG_BENCH_QUERIES_DIR/Resources/opennlp-tools-1.5.3.jar:${BIG_BENCH_QUERIES_DIR}/Resources/bigbenchqueriesmr.jar"

        hive $HIVE_AUXPATH $HIVE_PARAMS -i "$COMBINED_PARAMS_FILE" -f "$HIVE_SCRIPT"
}
