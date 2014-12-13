#!/usr/bin/env bash

query_run_main_method () {
	QUERY1_SCRIPT="$QUERY_DIR/hive1.sql"
	if [ ! -r "$QUERY1_SCRIPT" ]
	then
		echo "SQL file $QUERY1_SCRIPT can not be read."
		exit 1
	fi

	QUERY2_SCRIPT="$QUERY_DIR/hive2.sql"
	if [ ! -r "$QUERY2_SCRIPT" ]
	then
		echo "SQL file $QUERY2_SCRIPT can not be read."
		exit 1
	fi

	#EXECUTION Plan:
	#step 1.	hadoop fs	:	Copying LinearRegression jar to HDFS
	#step 2.	hive1.sql	:	Settting up intermediate views and tables
	#step 3.	hadoop m/r	:	Running LinearRegression on tables
	#step 4.	hive2.sql	:	Combining output into 1 table
	#step 5. 	hadoop fs	: 	Cleaning up

	MATRIX_MAX=10

	MR_JAR="${BIG_BENCH_QUERIES_DIR}/Resources/bigbenchqueriesmr.jar"
	MR_CLASS="de.bankmark.bigbench.queries.${QUERY_NAME}.MRlinearRegression"
	MR_JARCLASS="${MR_JAR} ${MR_CLASS}"

	# define used temp tables
	MATRIX_BASENAME="${TABLE_PREFIX}_matrix"
	MATRIX_BASEDIR="$TEMP_DIR/$MATRIX_BASENAME"
	LM_BASENAME="${TABLE_PREFIX}_lm"
	LM_BASEDIR="$TEMP_DIR/output"

	BINARY_PARAMS="$BINARY_PARAMS --hiveconf MATRIX_BASENAME=$MATRIX_BASENAME --hiveconf MATRIX_BASEDIR=$MATRIX_BASEDIR --hiveconf LM_BASENAME=$LM_BASENAME --hiveconf LM_BASEDIR=$LM_BASEDIR"

	echo "========================="
	echo "$QUERY_NAME Settings"
	echo "========================="
	echo "QUERY_DIR     $QUERY_DIR"
	echo "TEMP_DIR      $TEMP_DIR"
	echo "MR_JARCLASS   $MR_JARCLASS"

	#Step 1. Hadoop Part 0-----------------------------------------------------------------------
	# Copying jar to hdfs
	if [[ -z "$DEBUG_QUERY_PART" || $DEBUG_QUERY_PART -eq 1 ]] ; then
		echo "========================="
		echo "$QUERY_NAME Step 1: Prepare required resources"
		echo "========================="
		hadoop fs -copyFromLocal "${MR_JAR}" "${TEMP_DIR}/"
	fi

	#Step 2. Hive Part 1-----------------------------------------------------------------------
	if [[ -z "$DEBUG_QUERY_PART" || $DEBUG_QUERY_PART -eq 2 ]] ; then
		echo "========================="
		echo "$QUERY_NAME Step 2: exec hive query(s) part 1 (create matrix)"
		echo "========================="
		"$BINARY" $BINARY_PARAMS -i "$COMBINED_PARAMS_FILE" -f "$QUERY1_SCRIPT"
	fi

	#Step 3. Hadoop Part 1---- MRlinearRegression--------------------------------------------------------------
	if [[ -z "$DEBUG_QUERY_PART" || $DEBUG_QUERY_PART -eq 3 ]] ; then
		echo "========================="
		echo "$QUERY_NAME Step 3: prepare M/R job environment"
		echo "========================="
		hadoop fs -rm -r -skipTrash "$LM_BASEDIR"*

		echo "========================="
		echo "$QUERY_NAME Step 3: exec M/R job linear regression analysis"
		echo "========================="

		for (( i=1; i <= $MATRIX_MAX; i++ ))
		do
			echo "-------------------------"
			echo "$QUERY_NAME Step 3: linear regression analysis Matrix ${i}/10"
			echo "in: ${MATRIX_BASEDIR}${i}"
			echo "out: ${LM_BASEDIR}${i}"
			echo "Exec: hadoop jar ${MR_JARCLASS} ${MATRIX_BASEDIR}${i} ${LM_BASEDIR}${i}"
			echo "-------------------------"
			hadoop jar "${MR_JAR}" "${MR_CLASS}" "${MATRIX_BASEDIR}${i}" "${LM_BASEDIR}${i}" 
		done
		wait 
	fi

	#Step 4. Hive 2-----------------------------------------------------------------------
	if [[ -z "$DEBUG_QUERY_PART" || $DEBUG_QUERY_PART -eq 4 ]] ; then
		echo "========================="
		echo "$QUERY_NAME Step 4: exec hive query(s) part 2"
		echo "========================="
		"$BINARY" $BINARY_PARAMS -i "$COMBINED_PARAMS_FILE" -f "$QUERY2_SCRIPT"
	fi

	#Step 5. Hadoop  3-----------------------------------------------------------------------
	# Cleaning up
	if [[ -z "$DEBUG_QUERY_PART" || $DEBUG_QUERY_PART -eq 5 ]] ; then
		echo "========================="
		echo "$QUERY_NAME Step 5: cleaning up temporary files"
		echo "========================="
	
		hadoop fs -rm -r -skipTrash "${TEMP_DIR}"/*
	fi
}

query_run_clean_method () {
	"$BINARY" $BINARY_PARAMS -i "$COMBINED_PARAMS_FILE" -e "DROP TABLE IF EXISTS $TEMP_TABLE; DROP TABLE IF EXISTS $RESULT_TABLE;"
}
