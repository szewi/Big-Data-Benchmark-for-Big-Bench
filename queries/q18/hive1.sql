-- Global hive options (see: Big-Bench/setEnvVars)
set hive.exec.parallel=${env:BIG_BENCH_hive_exec_parallel};
set hive.exec.parallel.thread.number=${env:BIG_BENCH_hive_exec_parallel_thread_number};
set hive.exec.compress.intermediate=${env:BIG_BENCH_hive_exec_compress_intermediate};
set mapred.map.output.compression.codec=${env:BIG_BENCH_mapred_map_output_compression_codec};
set hive.exec.compress.output=${env:BIG_BENCH_hive_exec_compress_output};
set mapred.output.compression.codec=${env:BIG_BENCH_mapred_output_compression_codec};

--display settings
set hive.exec.parallel;
set hive.exec.parallel.thread.number;
set hive.exec.compress.intermediate;
set mapred.map.output.compression.codec;
set hive.exec.compress.output;
set mapred.output.compression.codec;

-- Database
use ${env:BIG_BENCH_HIVE_DATABASE};

-- Resources

-- Result file configuration
set QUERY_NUM=q18;
set QUERY_TMP_DIR=${env:BIG_BENCH_HDFS_ABSOLUTE_TEMP_DIR}/${hiveconf:QUERY_NUM}tmp;


--ss_sold_date_sk between 1999-09-02 and 1999-05-02: Days from 1900-01-01 till 1999-09-02 ==> 36403 
DROP VIEW IF EXISTS q18_time_series_store;
CREATE VIEW q18_time_series_store AS 
  SELECT 
    ss_store_sk AS store,
    ss_sold_date_sk AS d,
    sum(ss_net_paid) AS sales
  FROM store_sales
  WHERE ss_sold_date_sk > 36403 
        and ss_sold_date_sk < 36403+90
  GROUP BY ss_store_sk, ss_sold_date_sk;

--SELECT * FROM store_sales WHERE ss_store_sk =1 AND  ss_sold_date_sk > 35840 and ss_sold_date_sk < 35840+90;


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



CREATE TABLE q18_matrix1 ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:QUERY_TMP_DIR}/q18_matrix1'
  AS SELECT d, sales FROM q18_time_series_store WHERE store = 1;

CREATE TABLE q18_matrix2 ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:QUERY_TMP_DIR}/q18_matrix2'
  AS SELECT d, sales FROM q18_time_series_store WHERE store = 2;

CREATE TABLE q18_matrix3 ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:QUERY_TMP_DIR}/q18_matrix3'
  AS SELECT d, sales FROM q18_time_series_store WHERE store = 3;

CREATE TABLE q18_matrix4 ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:QUERY_TMP_DIR}/q18_matrix4'
  AS SELECT d, sales FROM q18_time_series_store WHERE store = 4;

CREATE TABLE q18_matrix5 ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:QUERY_TMP_DIR}/q18_matrix5'
  AS SELECT d, sales FROM q18_time_series_store WHERE store = 5;

CREATE TABLE q18_matrix6 ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:QUERY_TMP_DIR}/q18_matrix6'
  AS SELECT d, sales FROM q18_time_series_store WHERE store = 6;

CREATE TABLE q18_matrix7 ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:QUERY_TMP_DIR}/q18_matrix7'
  AS SELECT d, sales FROM q18_time_series_store WHERE store = 7;

CREATE TABLE q18_matrix8 ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:QUERY_TMP_DIR}/q18_matrix8'
  AS SELECT d, sales FROM q18_time_series_store WHERE store = 8;

CREATE TABLE q18_matrix9 ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:QUERY_TMP_DIR}/q18_matrix9'
  AS SELECT d, sales FROM q18_time_series_store WHERE store = 9;

CREATE TABLE q18_matrix10 ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:QUERY_TMP_DIR}/q18_matrix10'
  AS SELECT d, sales FROM q18_time_series_store WHERE store = 10;

CREATE TABLE q18_matrix11 ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:QUERY_TMP_DIR}/q18_matrix11'
  AS SELECT d, sales FROM q18_time_series_store WHERE store = 11;

CREATE TABLE q18_matrix12 ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:QUERY_TMP_DIR}/q18_matrix12'
  AS SELECT d, sales FROM q18_time_series_store WHERE store = 12;


DROP VIEW IF EXISTS q18_time_series_store;

