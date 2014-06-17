#!/usr/bin/env bash

ENV_SETTINGS="`dirname $0`/../../setEnvVars"
if [ ! -f "$ENV_SETTINGS" ]
then
        echo "Environment setup file $ENV_SETTINGS not found"
        exit 1
else
        source "$ENV_SETTINGS"
fi

logEnvInformation

#To debug start with: 
#> run.sh <step> 
#to execute only specified step

#EXECUTION Plan:
#step 1.	prepare		:	Copying LinearRegression jar to HDFS etc
#step 2.	hive1.sql	:	Settting up intermediate views and tables
#step 3.	hadoop m/r	:	Running LinearRegression on tables
#step 4.	hive2.sql	:	Combining output into 1 table
#step 5.	hive3.sql	:	Combine output with sentiment analysis
#step 6. 	hadoop fs	: 	Cleaning up

MATRIX_MAX=12

QUERY_NUM="q18"
QUERY_DIR=${BIG_BENCH_QUERIES_DIR}/${QUERY_NUM}
QUERY_TMP_DIR=${BIG_BENCH_HDFS_ABSOLUTE_TEMP_DIR}/${QUERY_NUM}tmp
MR_JAR="${BIG_BENCH_QUERIES_DIR}/Resources/bigbenchqueriesmr.jar"
MR_CLASS="de.bankmark.bigbench.queries.${QUERY_NUM}.MRlinearRegression"
MR_JARCLASS="${MR_JAR} ${MR_CLASS}"

#Step 1. Haddop Part 0-----------------------------------------------------------------------
# Copying jar to hdfs
if [ $# -lt 1 ] || [ $1 -eq 1 ] ; then
	echo "========================="
	echo "$QUERY_NUM Step 1/6: Prepare required resources"
	echo "========================="
	hadoop fs -rm -r -skipTrash  "${QUERY_TMP_DIR}/*"
	hadoop fs -mkdir -p "${QUERY_TMP_DIR}"
	hadoop fs -chmod uga+rw  "${QUERY_TMP_DIR}"
	hadoop fs -copyFromLocal "${MR_JAR}" "${QUERY_TMP_DIR}/"
fi

#Step 2. Hive Part 1-----------------------------------------------------------------------
if [ $# -lt 1 ] || [ $1 -eq 2 ] ; then
	echo "========================="
	echo "$QUERY_NUM Step 2/6: exec hive query(s) part 1 (create matrix)"
	echo "========================="
	hive -f "${QUERY_DIR}/hive1.sql"
fi

#Step 3. Hadoop Part 1---- MRlinearRegression--------------------------------------------------------------
if [ $# -lt 1 ] || [ $1 -eq 3 ] ; then
	echo "========================="
	echo "$QUERY_NUM Step 3/6:  prepare M/R job environment"
	echo "========================="

	hadoop fs -rm -r -skipTrash  "${QUERY_TMP_DIR}/output*"

	echo "========================="
	echo "$QUERY_NUM Step 3/6: exec M/R job linear regression analysis"
	echo "========================="
	for (( i=1; i <= $MATRIX_MAX; i++ ))
	do
		echo "-------------------------"
		echo "$QUERY_NUM Step 3: linear regression analysis Matrix ${i}/12"
		echo "in: ${QUERY_TMP_DIR}/q18_matrix${i}"
		echo "out: ${QUERY_TMP_DIR}/output${i}"
		echo "Exec: hadoop jar ${MR_JARCLASS} ${QUERY_TMP_DIR}/q18_matrix${i} ${QUERY_TMP_DIR}/output${i} "
		echo "-------------------------"
		hadoop jar "${MR_JAR}" "${MR_CLASS}" "${QUERY_TMP_DIR}/q18_matrix${i}" "${QUERY_TMP_DIR}/output${i}"  
	done
fi

#Step 4. Hive 2-----------------------------------------------------------------------
if [ $# -lt 1 ] || [ $1 -eq 4 ] ; then
	echo "========================="
	echo "$QUERY_NUM Step 4/6: exec hive query(s) part 2, aggregate linear regression"
	echo "========================="
	hive -f "${QUERY_DIR}/hive2.sql"
fi

#Step 5. Hive 3-----------------------------------------------------------------------
if [ $# -lt 1 ] || [ $1 -eq 5 ] ; then
	echo "========================="
	echo "$QUERY_NUM Step 5/6: exec hive query(s) part 3, combine with sentiment analysis"
	echo "========================="
	echo "hive --auxpath \"$BIG_BENCH_QUERIES_DIR/Resources/opennlp-maxent-3.0.3.jar:$BIG_BENCH_QUERIES_DIR/Resources/opennlp-tools-1.5.3.jar\" -f \"${QUERY_DIR}/hive3.sql\""
	hive  --auxpath "$BIG_BENCH_QUERIES_DIR/Resources/opennlp-maxent-3.0.3.jar:$BIG_BENCH_QUERIES_DIR/Resources/opennlp-tools-1.5.3.jar" -f "${QUERY_DIR}/hive3.sql"

fi

#Step 6. Hadoop  3-----------------------------------------------------------------------
# Cleaning up
if [ $# -lt 1 ] || [ $1 -eq 6 ] ; then
	echo "========================="
	echo "$QUERY_NUM Step 6/6: cleaning up temporary files"
	echo "========================="
	hive  -f "${QUERY_DIR}/cleanup.sql"
	hadoop fs -rm -r -skipTrash  "${QUERY_TMP_DIR}/*"
fi

echo "======= $QUERY_NUM  result ======="
echo "results in: ${BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${QUERY_NUM}result"
echo "to display : hadoop fs -cat ${BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${QUERY_NUM}result/*"
echo "========================="
