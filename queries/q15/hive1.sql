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
DROP TABLE IF EXISTS q15_matrix1;
DROP TABLE IF EXISTS q15_matrix2;
DROP TABLE IF EXISTS q15_matrix3;
DROP TABLE IF EXISTS q15_matrix4;
DROP TABLE IF EXISTS q15_matrix5;
DROP TABLE IF EXISTS q15_matrix6;
DROP TABLE IF EXISTS q15_matrix7;
DROP TABLE IF EXISTS q15_matrix8;
DROP TABLE IF EXISTS q15_matrix9;
DROP TABLE IF EXISTS q15_matrix10;

----store time series------------------------------------------------------------------

CREATE TABLE q15_matrix1 (d BIGINT,sales DOUBLE)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:TEMP_DIR}/q15_matrix1'
;


CREATE TABLE q15_matrix2(d BIGINT,sales DOUBLE)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:TEMP_DIR}/q15_matrix2'
;


CREATE TABLE q15_matrix3(d BIGINT,sales DOUBLE)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:TEMP_DIR}/q15_matrix3'
;


CREATE TABLE q15_matrix4(d BIGINT,sales DOUBLE)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:TEMP_DIR}/q15_matrix4'
;


CREATE TABLE q15_matrix5(d BIGINT,sales DOUBLE)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:TEMP_DIR}/q15_matrix5'
;


CREATE TABLE q15_matrix6(d BIGINT,sales DOUBLE)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:TEMP_DIR}/q15_matrix6'
;


CREATE TABLE q15_matrix7(d BIGINT,sales DOUBLE)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:TEMP_DIR}/q15_matrix7'
;

CREATE TABLE q15_matrix8(d BIGINT,sales DOUBLE)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:TEMP_DIR}/q15_matrix8'
;


CREATE TABLE q15_matrix9(d BIGINT,sales DOUBLE)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:TEMP_DIR}/q15_matrix9'
;


CREATE TABLE q15_matrix10(d BIGINT,sales DOUBLE)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:TEMP_DIR}/q15_matrix10'
;

-- parallel fails for this task
set hive.exec.parallel=false;

FROM (
	SELECT  i.i_category_id AS cat, -- ranges from 1 to 10
		s.ss_sold_date_sk AS d,
		sum(s.ss_net_paid) AS sales
	FROM    store_sales s
	INNER JOIN item i ON s.ss_item_sk = i.i_item_sk 
	WHERE   i.i_category_id IS NOT NULL
		-- and date is within range ss_sold_date_sk > 1999-09-02  AND s.ss_sold_date_sk < 1999-09-02 + 365 days
		AND s.ss_sold_date_sk > 36403 AND s.ss_sold_date_sk < 36403+365
		-- and for a given store
		AND s.ss_store_sk = 10 -- ranges from 1 to 12

	GROUP BY i.i_category_id, s.ss_sold_date_sk
) tsc
INSERT OVERWRITE TABLE q15_matrix1 SELECT d, sales WHERE cat = 1
INSERT OVERWRITE TABLE q15_matrix2 SELECT d, sales WHERE cat = 2
INSERT OVERWRITE TABLE q15_matrix3 SELECT d, sales WHERE cat = 3
INSERT OVERWRITE TABLE q15_matrix4 SELECT d, sales WHERE cat = 4
INSERT OVERWRITE TABLE q15_matrix5 SELECT d, sales WHERE cat = 5
INSERT OVERWRITE TABLE q15_matrix6 SELECT d, sales WHERE cat = 6
INSERT OVERWRITE TABLE q15_matrix7 SELECT d, sales WHERE cat = 7
INSERT OVERWRITE TABLE q15_matrix8 SELECT d, sales WHERE cat = 8
INSERT OVERWRITE TABLE q15_matrix9 SELECT d, sales WHERE cat = 9
INSERT OVERWRITE TABLE q15_matrix10 SELECT d, sales WHERE cat = 10
;



--cleaning up -------------------------------------------------
