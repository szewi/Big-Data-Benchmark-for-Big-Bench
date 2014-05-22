-- Global hive options (see: Big-Bench/setEnvVars)
--set hive.exec.parallel=${env:BIG_BENCH_hive_exec_parallel};
--set hive.exec.parallel.thread.number=${env:BIG_BENCH_hive_exec_parallel_thread_number};
--set hive.exec.compress.intermediate=${env:BIG_BENCH_hive_exec_compress_intermediate};
--set mapred.map.output.compression.codec=${env:BIG_BENCH_mapred_map_output_compression_codec};
--set hive.exec.compress.output=${env:BIG_BENCH_hive_exec_compress_output};
--set mapred.output.compression.codec=${env:BIG_BENCH_mapred_output_compression_codec};
--set hive.default.fileformat=${env:BIG_BENCH_hive_default_fileformat};
--set hive.optimize.mapjoin.mapreduce=${env:BIG_BENCH_hive_optimize_mapjoin_mapreduce};
--set hive.optimize.bucketmapjoin=${env:BIG_BENCH_hive_optimize_bucketmapjoin};
--set hive.optimize.bucketmapjoin.sortedmerge=${env:BIG_BENCH_hive_optimize_bucketmapjoin_sortedmerge};
--set hive.auto.convert.join=${env:BIG_BENCH_hive_auto_convert_join};
--set hive.auto.convert.sortmerge.join=${env:BIG_BENCH_hive_auto_convert_sortmerge_join};
--set hive.auto.convert.sortmerge.join.noconditionaltask=${env:BIG_BENCH_hive_auto_convert_sortmerge_join_noconditionaltask};
--set hive.optimize.ppd=${env:BIG_BENCH_hive_optimize_ppd};
--set hive.optimize.index.filter=${env:BIG_BENCH_hive_optimize_index_filter};

--display settings
set hive.exec.parallel;
set hive.exec.parallel.thread.number;
set hive.exec.compress.intermediate;
set mapred.map.output.compression.codec;
set hive.exec.compress.output;
set mapred.output.compression.codec;
set hive.default.fileformat;
set hive.optimize.mapjoin.mapreduce;
set hive.mapjoin.smalltable.filesize;
set hive.optimize.bucketmapjoin;
set hive.optimize.bucketmapjoin.sortedmerge;
set hive.auto.convert.join;
set hive.auto.convert.sortmerge.join;
set hive.auto.convert.sortmerge.join.noconditionaltask;
set hive.optimize.ppd;
set hive.optimize.index.filter;

-- Database
use ${env:BIG_BENCH_HIVE_DATABASE};

-- Resources

-- Result file configuration
set QUERY_NUM=q18;
set QUERY_TMP_DIR=${env:BIG_BENCH_HDFS_ABSOLUTE_TEMP_DIR}/${hiveconf:QUERY_NUM}tmp;


DROP TABLE IF EXISTS q18_matrix1;
DROP TABLE IF EXISTS q18_matrix2;
DROP TABLE IF EXISTS q18_matrix3;
DROP TABLE IF EXISTS q18_matrix4;
DROP TABLE IF EXISTS q18_matrix5;
DROP TABLE IF EXISTS q18_matrix6;
DROP TABLE IF EXISTS q18_matrix7;
DROP TABLE IF EXISTS q18_matrix8;
DROP TABLE IF EXISTS q18_matrix9;
DROP TABLE IF EXISTS q18_matrix10;
DROP TABLE IF EXISTS q18_matrix11;
DROP TABLE IF EXISTS q18_matrix12;



CREATE TABLE q18_matrix1 (d BIGINT, sales BIGINT)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:QUERY_TMP_DIR}/q18_matrix1'
;

CREATE TABLE q18_matrix2 (d BIGINT, sales BIGINT)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:QUERY_TMP_DIR}/q18_matrix2'
;

CREATE TABLE q18_matrix3 (d BIGINT, sales BIGINT)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:QUERY_TMP_DIR}/q18_matrix3'
;

CREATE TABLE q18_matrix4 (d BIGINT, sales BIGINT)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:QUERY_TMP_DIR}/q18_matrix4'
;

CREATE TABLE q18_matrix5 (d BIGINT, sales BIGINT)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:QUERY_TMP_DIR}/q18_matrix5'
;

CREATE TABLE q18_matrix6 (d BIGINT, sales BIGINT)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:QUERY_TMP_DIR}/q18_matrix6'
;

CREATE TABLE q18_matrix7 (d BIGINT, sales BIGINT)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:QUERY_TMP_DIR}/q18_matrix7'
;

CREATE TABLE q18_matrix8 (d BIGINT, sales BIGINT)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:QUERY_TMP_DIR}/q18_matrix8'
;

CREATE TABLE q18_matrix9 (d BIGINT, sales BIGINT)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:QUERY_TMP_DIR}/q18_matrix9'
;

CREATE TABLE q18_matrix10 (d BIGINT, sales BIGINT)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:QUERY_TMP_DIR}/q18_matrix10'
;

CREATE TABLE q18_matrix11 (d BIGINT, sales BIGINT)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:QUERY_TMP_DIR}/q18_matrix11'
;

CREATE TABLE q18_matrix12 (d BIGINT, sales BIGINT)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:QUERY_TMP_DIR}/q18_matrix12'
;


--ss_sold_date_sk between 1999-09-02 and 1999-05-02: Days from 1900-01-01 till 1999-09-02 ==> 36403 
FROM (
  SELECT 
    ss_store_sk AS store,
    ss_sold_date_sk AS d,
    sum(ss_net_paid) AS sales
  FROM store_sales
  WHERE ss_sold_date_sk > 36403 
        and ss_sold_date_sk < 36403+90
  GROUP BY ss_store_sk, ss_sold_date_sk
) tmp
INSERT OVERWRITE TABLE q18_matrix1 SELECT d, sales WHERE store = 1
INSERT OVERWRITE TABLE q18_matrix2 SELECT d, sales  WHERE store = 2
INSERT OVERWRITE TABLE q18_matrix3 SELECT d, sales  WHERE store = 3
INSERT OVERWRITE TABLE q18_matrix4 SELECT d, sales  WHERE store = 4
INSERT OVERWRITE TABLE q18_matrix5 SELECT d, sales  WHERE store = 5
INSERT OVERWRITE TABLE q18_matrix6 SELECT d, sales  WHERE store = 6
INSERT OVERWRITE TABLE q18_matrix7 SELECT d, sales  WHERE store = 7
INSERT OVERWRITE TABLE q18_matrix8 SELECT d, sales  WHERE store = 8
INSERT OVERWRITE TABLE q18_matrix9 SELECT d, sales  WHERE store = 9
INSERT OVERWRITE TABLE q18_matrix10 SELECT d, sales  WHERE store = 10
INSERT OVERWRITE TABLE q18_matrix11 SELECT d, sales  WHERE store = 11
INSERT OVERWRITE TABLE q18_matrix12 SELECT d, sales  WHERE store = 12
;
--SELECT * FROM store_sales WHERE ss_store_sk =1 AND  ss_sold_date_sk > 35840 and ss_sold_date_sk < 35840+90;




