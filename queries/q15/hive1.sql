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
set QUERY_NUM=q15;
set QUERY_TMP_DIR=${env:BIG_BENCH_HDFS_ABSOLUTE_TEMP_DIR}/${hiveconf:QUERY_NUM}tmp;

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
DROP TABLE IF EXISTS q15_time_series_category;

CREATE TABLE IF NOT EXISTS q15_time_series_category AS
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

GROUP BY i.i_category_id, s.ss_sold_date_sk;
-------------------------------------------------------------------------------

  
CREATE TABLE q15_matrix1 
(
d BIGINT,
sales DOUBLE
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE 
LOCATION '${hiveconf:QUERY_TMP_DIR}/q15_matrix1'
;


CREATE TABLE q15_matrix2
(
d BIGINT,
sales DOUBLE
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE 
LOCATION '${hiveconf:QUERY_TMP_DIR}/q15_matrix2'
;


CREATE TABLE q15_matrix3
(
d BIGINT,
sales DOUBLE
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE 
LOCATION '${hiveconf:QUERY_TMP_DIR}/q15_matrix3'
;


CREATE TABLE q15_matrix4
(
d BIGINT,
sales DOUBLE
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE 
LOCATION '${hiveconf:QUERY_TMP_DIR}/q15_matrix4'
;


CREATE TABLE q15_matrix5
(
d BIGINT,
sales DOUBLE
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE 
LOCATION '${hiveconf:QUERY_TMP_DIR}/q15_matrix5'
;


CREATE TABLE q15_matrix6
(
d BIGINT,
sales DOUBLE
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE 
LOCATION '${hiveconf:QUERY_TMP_DIR}/q15_matrix6'
;


CREATE TABLE q15_matrix7
(
d BIGINT,
sales DOUBLE
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE 
LOCATION '${hiveconf:QUERY_TMP_DIR}/q15_matrix7'
;

CREATE TABLE q15_matrix8
(
d BIGINT,
sales DOUBLE
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE 
LOCATION '${hiveconf:QUERY_TMP_DIR}/q15_matrix8'
;


CREATE TABLE q15_matrix9
(
d BIGINT,
sales DOUBLE
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE 
LOCATION '${hiveconf:QUERY_TMP_DIR}/q15_matrix9'
;


CREATE TABLE q15_matrix10
(
d BIGINT,
sales DOUBLE
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE 
LOCATION '${hiveconf:QUERY_TMP_DIR}/q15_matrix10'
;

-- parallel fails for this task
set hive.exec.parallel=false;

FROM q15_time_series_category tsc
INSERT INTO TABLE q15_matrix1 SELECT d, sales WHERE cat = 1
INSERT INTO TABLE q15_matrix2 SELECT d, sales WHERE cat = 2
INSERT INTO TABLE q15_matrix3 SELECT d, sales WHERE cat = 3
INSERT INTO TABLE q15_matrix4 SELECT d, sales WHERE cat = 4
INSERT INTO TABLE q15_matrix5 SELECT d, sales WHERE cat = 5
INSERT INTO TABLE q15_matrix6 SELECT d, sales WHERE cat = 6
INSERT INTO TABLE q15_matrix7 SELECT d, sales WHERE cat = 7
INSERT INTO TABLE q15_matrix8 SELECT d, sales WHERE cat = 8
INSERT INTO TABLE q15_matrix9 SELECT d, sales WHERE cat = 9
INSERT INTO TABLE q15_matrix10 SELECT d, sales WHERE cat = 10
;



--cleaning up -------------------------------------------------
DROP TABLE IF EXISTS q15_time_series_category;


