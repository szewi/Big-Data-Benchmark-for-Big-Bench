#!/usr/bin/env bash


#To debug start with: 
#> run.sh <step> 
#to execute only the specified step

#EXECUTION Plan:
#step 1.  rm/mkdir MH_TMP_DIR	:	hadoop fs rm/mkdir MH_TMP_DIR
#step 2.  hive q26.sql		:	Run hive querys to extract kmeans input data
#step 3.  mahout input		:	Generating sparse vectors
#step 4.  mahout kmeans		:	Calculating k-means"
#step 5.  mahout dump > hdfs/res:	Converting result and copy result do hdfs query result folder
#step 6.  hive && hdfs 		:	cleanup.sql && hadoop fs rm MH


QUERY_NUM="q26"
QUERY_DIR="${BIG_BENCH_QUERIES_DIR}/${QUERY_NUM}"
MH_TMP_DIR=$BIG_BENCH_HDFS_ABSOLUTE_TEMP_DIR/$QUERY_NUM
HIVE_TABLE_NAME=${MH_TMP_DIR}/ctable

resultTableName=${QUERY_NUM}result
HDFS_RESULT_DIR=${BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${resultTableName}

if [ $# -lt 1 ] || [ $1 -eq 1 ] ; then
	echo "========================="
	echo "$QUERY_NUM Step 1/6: Prepare temp dir"
	echo "========================="
	hadoop fs -rm -r "$MH_TMP_DIR" &
	hadoop fs -rm -r "$HDFS_RESULT_DIR" &
	wait
	hadoop fs -mkdir -p "$MH_TMP_DIR" &
	hadoop fs -mkdir -p "$HDFS_RESULT_DIR" &
	wait

fi


if [ $# -lt 1 ] || [ $1 -eq 2 ] ; then
	echo "========================="
	echo "$QUERY_NUM Step 2/6: Executing hive queries"
	echo "tmp output: ${HIVE_TABLE_NAME}"
	echo "========================="
	# Write input for k-means into ctable
	hive -hiveconf "MH_TMP_DIR=${HIVE_TABLE_NAME}" -f "${QUERY_DIR}"/${QUERY_NUM}.sql
fi


if [ $# -lt 1 ] || [ $1 -eq 3 ] ; then
	echo "========================="
	echo "$QUERY_NUM Step 3/6: Generating sparse vectors"
	echo "Command "mahout org.apache.mahout.clustering.conversion.InputDriver -i "${HIVE_TABLE_NAME}" -o "${MH_TMP_DIR}"/Vec -v org.apache.mahout.math.RandomAccessSparseVector #-c UTF-8 
	echo "tmp output: ${MH_TMP_DIR}/Vec"
	echo "========================="

	mahout org.apache.mahout.clustering.conversion.InputDriver -i "${HIVE_TABLE_NAME}" -o "${MH_TMP_DIR}"/Vec -v org.apache.mahout.math.RandomAccessSparseVector #-c UTF-8 
fi



if [ $# -lt 1 ] || [ $1 -eq 4 ] ; then
	echo "========================="
	echo "$QUERY_NUM Step 4/6: Calculating k-means"
	echo "Command "mahout kmeans -i "$MH_TMP_DIR"/Vec -c "$MH_TMP_DIR"/init-clusters -o "$MH_TMP_DIR"/kmeans-clusters -dm org.apache.mahout.common.distance.CosineDistanceMeasure -x 10 -k 8 -ow -cl
	echo "tmp output: $MH_TMP_DIR/kmeans-clusters"
	echo "========================="

	mahout kmeans -i "$MH_TMP_DIR"/Vec -c "$MH_TMP_DIR"/init-clusters -o "$MH_TMP_DIR"/kmeans-clusters -dm org.apache.mahout.common.distance.CosineDistanceMeasure -x 10 -k 8 -ow -cl
fi


if [ $# -lt 1 ] || [ $1 -eq 5 ] ; then
	echo "========================="
	echo "$QUERY_NUM Step 5/6: Converting result and store in hdfs ${HDFS_RESULT_DIR}/cluster.txt"
	echo "command: mahout clusterdump -i $MH_TMP_DIR/kmeans-clusters/clusters-*-final  -dm org.apache.mahout.common.distance.CosineDistanceMeasure -of TEXT | hadoop fs -copyFromLocal - ${HDFS_RESULT_DIR}/cluster.txt"
	echo "========================="
	
	mahout clusterdump -i "$MH_TMP_DIR"/kmeans-clusters/clusters-*-final  -dm org.apache.mahout.common.distance.CosineDistanceMeasure -of TEXT | hadoop fs -copyFromLocal - "${HDFS_RESULT_DIR}"/cluster.txt
	#mahout seqdump -i $MH_TMP_DIR/Vec/ -c $MH_TMP_DIR/kmeans-clusters -o $MH_TMP_DIR/results -dm org.apache.mahout.common.distance.CosineDistanceMeasure -x 10 -k 8 -ow -cl
fi


if [ $# -lt 1 ] || [ $1 -eq 6 ] ; then
	echo "========================="
	echo "$QUERY_NUM Step 6/6: Clean up"
	echo "========================="
	hive -f "${QUERY_DIR}"/cleanup.sql
	hadoop fs -rm -r "$MH_TMP_DIR"
fi


echo "======= $QUERY_NUM  result ======="
echo "results in: ${BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${QUERY_NUM}result"
echo "to display : hadoop fs -cat ${BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${QUERY_NUM}result/*"
echo "========================="



