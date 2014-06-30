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
DROP TABLE IF EXISTS ${hiveconf:MATRIX_BASENAME}1;
DROP TABLE IF EXISTS ${hiveconf:MATRIX_BASENAME}2;
DROP TABLE IF EXISTS ${hiveconf:MATRIX_BASENAME}3;
DROP TABLE IF EXISTS ${hiveconf:MATRIX_BASENAME}4;
DROP TABLE IF EXISTS ${hiveconf:MATRIX_BASENAME}5;
DROP TABLE IF EXISTS ${hiveconf:MATRIX_BASENAME}6;
DROP TABLE IF EXISTS ${hiveconf:MATRIX_BASENAME}7;
DROP TABLE IF EXISTS ${hiveconf:MATRIX_BASENAME}8;
DROP TABLE IF EXISTS ${hiveconf:MATRIX_BASENAME}9;
DROP TABLE IF EXISTS ${hiveconf:MATRIX_BASENAME}10;
DROP TABLE IF EXISTS ${hiveconf:MATRIX_BASENAME}11;
DROP TABLE IF EXISTS ${hiveconf:MATRIX_BASENAME}12;


CREATE TABLE ${hiveconf:MATRIX_BASENAME}1 (d BIGINT, sales BIGINT)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:MATRIX_BASEDIR}1'
;

CREATE TABLE ${hiveconf:MATRIX_BASENAME}2 (d BIGINT, sales BIGINT)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:MATRIX_BASEDIR}2'
;

CREATE TABLE ${hiveconf:MATRIX_BASENAME}3 (d BIGINT, sales BIGINT)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:MATRIX_BASEDIR}3'
;

CREATE TABLE ${hiveconf:MATRIX_BASENAME}4 (d BIGINT, sales BIGINT)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:MATRIX_BASEDIR}4'
;

CREATE TABLE ${hiveconf:MATRIX_BASENAME}5 (d BIGINT, sales BIGINT)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:MATRIX_BASEDIR}5'
;

CREATE TABLE ${hiveconf:MATRIX_BASENAME}6 (d BIGINT, sales BIGINT)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:MATRIX_BASEDIR}6'
;

CREATE TABLE ${hiveconf:MATRIX_BASENAME}7 (d BIGINT, sales BIGINT)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:MATRIX_BASEDIR}7'
;

CREATE TABLE ${hiveconf:MATRIX_BASENAME}8 (d BIGINT, sales BIGINT)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:MATRIX_BASEDIR}8'
;

CREATE TABLE ${hiveconf:MATRIX_BASENAME}9 (d BIGINT, sales BIGINT)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:MATRIX_BASEDIR}9'
;

CREATE TABLE ${hiveconf:MATRIX_BASENAME}10 (d BIGINT, sales BIGINT)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:MATRIX_BASEDIR}10'
;

CREATE TABLE ${hiveconf:MATRIX_BASENAME}11 (d BIGINT, sales BIGINT)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:MATRIX_BASEDIR}11'
;

CREATE TABLE ${hiveconf:MATRIX_BASENAME}12 (d BIGINT, sales BIGINT)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:MATRIX_BASEDIR}12'
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
INSERT OVERWRITE TABLE ${hiveconf:MATRIX_BASENAME}1 SELECT d, sales WHERE store = 1
INSERT OVERWRITE TABLE ${hiveconf:MATRIX_BASENAME}2 SELECT d, sales WHERE store = 2
INSERT OVERWRITE TABLE ${hiveconf:MATRIX_BASENAME}3 SELECT d, sales WHERE store = 3
INSERT OVERWRITE TABLE ${hiveconf:MATRIX_BASENAME}4 SELECT d, sales WHERE store = 4
INSERT OVERWRITE TABLE ${hiveconf:MATRIX_BASENAME}5 SELECT d, sales WHERE store = 5
INSERT OVERWRITE TABLE ${hiveconf:MATRIX_BASENAME}6 SELECT d, sales WHERE store = 6
INSERT OVERWRITE TABLE ${hiveconf:MATRIX_BASENAME}7 SELECT d, sales WHERE store = 7
INSERT OVERWRITE TABLE ${hiveconf:MATRIX_BASENAME}8 SELECT d, sales WHERE store = 8
INSERT OVERWRITE TABLE ${hiveconf:MATRIX_BASENAME}9 SELECT d, sales WHERE store = 9
INSERT OVERWRITE TABLE ${hiveconf:MATRIX_BASENAME}10 SELECT d, sales WHERE store = 10
INSERT OVERWRITE TABLE ${hiveconf:MATRIX_BASENAME}11 SELECT d, sales WHERE store = 11
INSERT OVERWRITE TABLE ${hiveconf:MATRIX_BASENAME}12 SELECT d, sales WHERE store = 12
;
--SELECT * FROM store_sales WHERE ss_store_sk =1 AND  ss_sold_date_sk > 35840 and ss_sold_date_sk < 35840+90;
