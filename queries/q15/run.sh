#!/usr/bin/env bash

ENV_SETTINGS="`dirname $0`/../../setEnvVars"
if [ ! -f "$ENV_SETTINGS" ]
then
        echo "Environment setup file $ENV_SETTINGS not found"
        exit 1
else
        source "$ENV_SETTINGS"
fi

#To debug start with: 
#> run.sh <step> 
#to execute only specified step

#EXECUTION Plan:
#step 1.	hadoop fs	:	Copying LinearRegression jar to HDFS
#step 2.	hive1.sql	:	Settting up intermediate views and tables
#step 3.	hadoop m/r	:	Running LinearRegression on tables
#step 4.	hive2.sql	:	Combining output into 1 table
#step 5. 	hadoop fs	: 	Cleaning up

MATRIX_MAX=10
QUERY_NUM="q15"
QUERY_DIR=${BIG_BENCH_QUERIES_DIR}/${QUERY_NUM}
QUERY_TMP_DIR=${BIG_BENCH_HDFS_ABSOLUTE_TEMP_DIR}/${QUERY_NUM}tmp
MR_JAR="${BIG_BENCH_QUERIES_DIR}/Resources/bigbenchqueriesmr.jar"
MR_CLASS="de.bankmark.bigbench.queries.${QUERY_NUM}.MRlinearRegression"
MR_JARCLASS="${MR_JAR} ${MR_CLASS}"

echo "========================="
echo "$QUERY_NUM Settings"
echo "========================="
echo "QUERY_DIR     $QUERY_DIR"
echo "QUERY_TMP_DIR $QUERY_TMP_DIR"
echo "MR_JARCLASS   $MR_JARCLASS"

#Step 1. Haddop Part 0-----------------------------------------------------------------------
# Copying jar to hdfs
if [ $# -lt 1 ] || [ $1 -eq 1 ] ; then
	echo "========================="
	echo "$QUERY_NUM Step 1: Prepare required resources"
	echo "========================="
	hadoop fs -rm -r -skipTrash  "${QUERY_TMP_DIR}/*"
	hadoop fs -mkdir -p "${QUERY_TMP_DIR}"
	hadoop fs -chmod uga+rw  "${QUERY_TMP_DIR}"
	hadoop fs -copyFromLocal "${MR_JAR}" "${QUERY_TMP_DIR}/"
fi

#Step 2. Hive Part 1-----------------------------------------------------------------------
if [ $# -lt 1 ] || [ $1 -eq 2 ] ; then
	echo "========================="
	echo "$QUERY_NUM Step 2: exec hive query(s) part 1 (create matrix)"
	echo "========================="
	hive -f ${QUERY_DIR}/hive1.sql
fi

#Step 3. Hadoop Part 1---- MRlinearRegression--------------------------------------------------------------
if [ $# -lt 1 ] || [ $1 -eq 3 ] ; then
	echo "========================="
	echo "$QUERY_NUM Step 3:  prepare M/R job environment"
	echo "========================="
	hadoop fs -rm -r -skipTrash  ${QUERY_TMP_DIR}/output*

	echo "========================="
	echo "$QUERY_NUM Step 3: exec M/R job linear regression analysis"
	echo "========================="

	for (( i=1; i <= $MATRIX_MAX; i++ ))
	do
		echo "-------------------------"
		echo "$QUERY_NUM Step 3: linear regression analysis Matrix ${i}/10"
		echo "in: ${QUERY_TMP_DIR}/q15_matrix${i}"
		echo "out: ${QUERY_TMP_DIR}/output${i}"
		echo "Exec: hadoop jar ${MR_JARCLASS} ${QUERY_TMP_DIR}/q15_matrix${i} ${QUERY_TMP_DIR}/output${i} "
		echo "-------------------------"
		hadoop jar "${MR_JAR}" "${MR_CLASS}" "${QUERY_TMP_DIR}/q15_matrix${i}" "${QUERY_TMP_DIR}/output${i}" 
	done
	wait 
fi

#Step 4. Hive 2-----------------------------------------------------------------------
if [ $# -lt 1 ] || [ $1 -eq 4 ] ; then
	echo "========================="
	echo "$QUERY_NUM Step 4: exec hive query(s) part 2"
	echo "========================="
	hive -f ${QUERY_DIR}/hive2.sql
fi

#Step 5. Hadoop  3-----------------------------------------------------------------------
# Cleaning up
if [ $# -lt 1 ] || [ $1 -eq 5 ] ; then
	echo "========================="
	echo "$QUERY_NUM Step 5: cleaning up temporary files"
	echo "========================="
	
	hadoop fs -rm -r -skipTrash  ${QUERY_TMP_DIR}/*
fi

echo "======= $QUERY_NUM  result ======="
echo "results in: ${BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${QUERY_NUM}result"
echo "to display : hadoop fs -cat ${BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${QUERY_NUM}result/*"
echo "========================="
