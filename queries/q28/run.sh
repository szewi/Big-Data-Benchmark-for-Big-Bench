#!/usr/bin/env bash

QUERY_NUM="q28"
QUERY_DIR="${BIG_BENCH_QUERIES_DIR}/${QUERY_NUM}"
MH_TMP_DIR=$BIG_BENCH_HDFS_ABSOLUTE_TEMP_DIR/$QUERY_NUM
HIVE_TABLE_NAME=${MH_TMP_DIR}/ctable

resultTableName=${QUERY_NUM}result
HDFS_RESULT_DIR=${BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${resultTableName}


if [ $# -lt 1 ] || [ $1 -eq 1 ] ; then
	echo "========================="
	echo "$QUERY_NUM step 1/8: prepare/initialize"
	echo "========================="
	hadoop fs -rm -r "$MH_TMP_DIR"
	hadoop fs -mkdir "$MH_TMP_DIR"
fi


if [ $# -lt 1 ] || [ $1 -eq 2 ] ; then
	echo "========================="
	echo "$QUERY_NUM step 2/8: Executing hive queries"
	echo "tmp result in" ${MH_TMP_DIR}/ctable
	echo "tmp result in" ${MH_TMP_DIR}/ctable2
	echo "========================="

	# Write input for k-means into ctable
	hive -hiveconf "MH_TMP_DIR=${MH_TMP_DIR}/ctable" -f ${QUERY_DIR}/q28.sql
fi

#SEQ_FILE_1="$MH_TMP_DIR"/ctable
#SEQ_FILE_2="$MH_TMP_DIR"/ctable2
SEQ_FILE_1="$MH_TMP_DIR"/Seq1
SEQ_FILE_2="$MH_TMP_DIR"/Seq2
VEC_FILE_1="$MH_TMP_DIR"/Vec1
VEC_FILE_2="$MH_TMP_DIR"/Vec2

if [ $# -lt 1 ] || [ $1 -eq 3 ] ; then
	echo "========================="
	echo "$QUERY_NUM step 3/8: Generating sequence files"
	echo "Used Command: "mahout seqdirectory -i "$MH_TMP_DIR"/ctable -o "$MH_TMP_DIR"/Seq1 -ow
	echo "Used Command: "mahout seqdirectory -i "$MH_TMP_DIR"/ctable2 -o "$MH_TMP_DIR"/Seq2 -ow
	echo "tmp result in:" "$MH_TMP_DIR"/Seq1
	echo "tmp result in:" "$MH_TMP_DIR"/Seq2
	echo "========================="
	#mahout seqdirectory -i "$MH_TMP_DIR"/ctable -o "$SEQ_FILE_1" -ow
	#mahout seqdirectory -i "$MH_TMP_DIR"/ctable2 -o "$SEQ_FILE_2" -ow
fi


if [ $# -lt 1 ] || [ $1 -eq 4 ] ; then
	echo "========================="
	echo "$QUERY_NUM step 4/8: Generating vectors"
	echo "Used Command: "mahout seq2sparse -i "$SEQ_FILE_1" -o "$VEC_FILE_1" -seq -ow 
	echo "Used Command: "mahout seq2sparse -i "$SEQ_FILE_2" -o "$VEC_FILE_2" -seq -ow 
	echo "tmp result in: $VEC_FILE_1" 
	echo "tmp result in: $VEC_FILE_2"
	echo "========================="
	#mahout seq2sparse -i "$SEQ_FILE_1" -o "$VEC_FILE_1"  -ow #-lnorm -nv -wt tfidf
	#mahout seq2sparse -i "$SEQ_FILE_2" -o "$VEC_FILE_2"  -ow #-lnorm -nv -wt tfidf
fi



if [ $# -lt 1 ] || [ $1 -eq 5 ] ; then
	echo "========================="
	echo "$QUERY_NUM step 5/8: Training Classifier"
	echo "Used Command: "mahout trainnb -i "$VEC_FILE_1"/tfidf-vectors -o $MH_TMP_DIR/model -l key,rating,item,content -ow
	echo "tmp result in: " $MH_TMP_DIR/model
	echo "========================="
	#(pr_review_sk INT, pr_rating INT, pr_item_sk INT, pr_review_content STRING) 
	#mahout trainnb -i "$VEC_FILE_1"/tfidf-vectors -o "$MH_TMP_DIR"/model -l key,rating,item,content -ow
	#mahout trainnb -i "$VEC_FILE_1"/tfidf-vectors  -o  "$MH_TMP_DIR"/model -el -li "$MH_TMP_DIR"/labelindex -ow 
fi


if [ $# -lt 1 ] || [ $1 -eq 6 ] ; then
	echo "========================="
	echo "$QUERY_NUM step 6/8: Testing Classifier"
	echo "Used Command: "mahout testnb -i "$VEC_FILE_2"/tfidf-vectors -­m $MH_TMP_DIR/model -l $MH_TMP_DIR/label -o $MH_TMP_DIR/result -ow
	echo "tmp result in: " $MH_TMP_DIR/result
	echo "========================="

	#mahout testnb -i "$VEC_FILE_2"/tfidf-vectors -­m "$MH_TMP_DIR"/model -l "$MH_TMP_DIR"/label -o "$MH_TMP_DIR"/result -ow
	#mahout testnb -i "$VEC_FILE_2"/tfidf-vectors -m "$MH_TMP_DIR"/model -l "$MH_TMP_DIR"/labelindex -ow -o  "$MH_TMP_DIR"/result 
fi


if [ $# -lt 1 ] || [ $1 -eq 7 ] ; then
	echo "========================="
	echo "$QUERY_NUM step 7/8: write result to hdfs"
	echo "=========================
	#TODO copy result to ${BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${QUERY_NUM}result"

	#mahout seqdumper -i [FILE_NAME]
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

