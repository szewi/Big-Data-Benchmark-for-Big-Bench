#!/usr/bin/env bash

RESULT_TABLE1="${RESULT_TABLE}1"
RESULT_DIR1="$RESULT_DIR/$RESULT_TABLE1"
RESULT_TABLE2="${RESULT_TABLE}2"
RESULT_DIR2="$RESULT_DIR/$RESULT_TABLE2"

BINARY_PARAMS="$BINARY_PARAMS --hiveconf RESULT_TABLE1=$RESULT_TABLE1 --hiveconf RESULT_DIR1=$RESULT_DIR1 --hiveconf RESULT_TABLE2=$RESULT_TABLE2 --hiveconf RESULT_DIR2=$RESULT_DIR2"

query_run_main_method () {

	QUERY1_SCRIPT="$QUERY_DIR/q23_1.sql"
	if [ ! -r "$QUERY1_SCRIPT" ]
	then
		echo "SQL file $QUERY1_SCRIPT can not be read."
		exit 1
	fi

	QUERY2_SCRIPT="$QUERY_DIR/q23_2.sql"
	if [ ! -r "$QUERY2_SCRIPT" ]
	then
		echo "SQL file $QUERY2_SCRIPT can not be read."
		exit 1
	fi

	QUERY3_SCRIPT="$QUERY_DIR/q23_3.sql"
	if [ ! -r "$QUERY3_SCRIPT" ]
	then
		echo "SQL file $QUERY3_SCRIPT can not be read."
		exit 1
	fi

	#hadoop fs -rm -r -skipTrash "${RESULT_DIR}"/*
	#hadoop fs -mkdir -p "${RESULT_DIR}"
	#hadoop fs -chmod uga+rw "${RESULT_DIR}"


if [[ -z "$DEBUG_QUERY_PART" || $DEBUG_QUERY_PART -eq 1 ]] ; then
	echo "========================="
	echo "$QUERY_NAME Step 1/4: make view"
	echo "========================="
	runEngineCmd -f "$QUERY1_SCRIPT"
fi

if [[ -z "$DEBUG_QUERY_PART" || $DEBUG_QUERY_PART -eq 2 ]] ; then
	echo "========================="
	echo "$QUERY_NAME Step 2/4: make result 1"
	echo "========================="
	runEngineCmd -f "$QUERY2_SCRIPT"
fi

if [[ -z "$DEBUG_QUERY_PART" || $DEBUG_QUERY_PART -eq 3 ]] ; then
	echo "========================="
	echo "$QUERY_NAME Step 3/4: make result 2"
	echo "========================="
	runEngineCmd -f "$QUERY3_SCRIPT"
fi

if [[ -z "$DEBUG_QUERY_PART" || $DEBUG_QUERY_PART -eq 4 ]] ; then
	echo "========================="
	echo "$QUERY_NAME Step 4/4: cleanup"
	echo "========================="
	runEngineCmd -f "${QUERY_DIR}/cleanup.sql"
fi
}

query_run_clean_method () {
	runEngineCmd -e "DROP VIEW IF EXISTS $TEMP_TABLE; DROP TABLE IF EXISTS $RESULT_TABLE1; DROP TABLE IF EXISTS $RESULT_TABLE2;"
}
