#!/usr/bin/env bash

query_run_main_method () {
	HIVE_SCRIPT="$QUERY_DIR/$QUERY_NAME.sql"
	if [ ! -r "$HIVE_SCRIPT" ]
	then
		echo "SQL file $HIVE_SCRIPT can not be read."
		exit 1
	fi

	# define used temp tables
	TEMP_TABLE1="${TABLE_PREFIX}_DateRange"
	TEMP_TABLE2="${TABLE_PREFIX}_tmp_sales_review"
	TEMP_TABLE3="${TABLE_PREFIX}_tmp_webSales_date"

	HIVE_PARAMS="$HIVE_PARAMS -hiveconf TEMP_TABLE1=$TEMP_TABLE1 -hiveconf TEMP_TABLE2=$TEMP_TABLE2 -hiveconf TEMP_TABLE3=$TEMP_TABLE3"

        hive $HIVE_PARAMS -i "$COMBINED_PARAMS_FILE" -f "$HIVE_SCRIPT"
}
