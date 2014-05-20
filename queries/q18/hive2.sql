-- Global hive options (see: Big-Bench/setEnvVars)
set hive.exec.parallel=${env:BIG_BENCH_hive_exec_parallel};
set hive.exec.parallel.thread.number=${env:BIG_BENCH_hive_exec_parallel_thread_number};
set hive.exec.compress.intermediate=${env:BIG_BENCH_hive_exec_compress_intermediate};
set mapred.map.output.compression.codec=${env:BIG_BENCH_mapred_map_output_compression_codec};
set hive.exec.compress.output=${env:BIG_BENCH_hive_exec_compress_output};
set mapred.output.compression.codec=${env:BIG_BENCH_mapred_output_compression_codec};
set hive.default.fileformat=${env:BIG_BENCH_hive_default_fileformat};
set hive.optimize.mapjoin.mapreduce=${env:BIG_BENCH_hive_optimize_mapjoin_mapreduce};
set hive.optimize.bucketmapjoin=${env:BIG_BENCH_hive_optimize_bucketmapjoin};
set hive.optimize.bucketmapjoin.sortedmerge=${env:BIG_BENCH_hive_optimize_bucketmapjoin_sortedmerge};
set hive.auto.convert.join=${env:BIG_BENCH_hive_auto_convert_join};
set hive.auto.convert.sortmerge.join=${env:BIG_BENCH_hive_auto_convert_sortmerge_join};
set hive.auto.convert.sortmerge.join.noconditionaltask=${env:BIG_BENCH_hive_auto_convert_sortmerge_join_noconditionaltask};
set hive.optimize.ppd=${env:BIG_BENCH_hive_optimize_ppd};
set hive.optimize.index.filter=${env:BIG_BENCH_hive_optimize_index_filter};

--display settings
set hive.exec.parallel;
set hive.exec.parallel.thread.number;
set hive.exec.compress.intermediate;
set mapred.map.output.compression.codec;
set hive.exec.compress.output;
set mapred.output.compression.codec;
set hive.default.fileformat;
set hive.optimize.mapjoin.mapreduce;
set hive.optimize.bucketmapjoin;
set hive.optimize.bucketmapjoin.sortedmerge;
set hive.auto.convert.join;
set hive.auto.convert.sortmerge.join;
set hive.auto.convert.sortmerge.join.noconditionaltask;
-- Database
use ${env:BIG_BENCH_HIVE_DATABASE};

-- Resources

-- Result file configuration
set QUERY_NUM=q18;
set QUERY_TMP_DIR=${env:BIG_BENCH_HDFS_ABSOLUTE_TEMP_DIR}/${hiveconf:QUERY_NUM}tmp;
set resultTableName=${hiveconf:QUERY_NUM}result;
set resultFile=${env:BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${hiveconf:resultTableName};


-------------------------------------------------------------------------------

DROP TABLE IF EXISTS q18_lm1;
DROP TABLE IF EXISTS q18_lm2;
DROP TABLE IF EXISTS q18_lm3;
DROP TABLE IF EXISTS q18_lm4;
DROP TABLE IF EXISTS q18_lm5;
DROP TABLE IF EXISTS q18_lm6;
DROP TABLE IF EXISTS q18_lm7;
DROP TABLE IF EXISTS q18_lm8;
DROP TABLE IF EXISTS q18_lm9;
DROP TABLE IF EXISTS q18_lm10;
DROP TABLE IF EXISTS q18_lm11;
DROP TABLE IF EXISTS q18_lm12;

CREATE EXTERNAL TABLE q18_lm1 (
    intercept            DOUBLE,
    slope                DOUBLE
 )
 ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
 STORED AS TEXTFILE LOCATION '${hiveconf:QUERY_TMP_DIR}/output1/'
;

CREATE EXTERNAL TABLE q18_lm2 (
    intercept            DOUBLE,
    slope                DOUBLE
 )
 ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
 STORED AS TEXTFILE LOCATION '${hiveconf:QUERY_TMP_DIR}/output2/'
;

CREATE EXTERNAL TABLE q18_lm3 (
    intercept            DOUBLE,
    slope                DOUBLE
 )
 ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
 STORED AS TEXTFILE LOCATION '${hiveconf:QUERY_TMP_DIR}/output3/'
;

CREATE EXTERNAL TABLE q18_lm4 (
    intercept            DOUBLE,
    slope                DOUBLE
 )
 ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
 STORED AS TEXTFILE LOCATION '${hiveconf:QUERY_TMP_DIR}/output4/'
;

CREATE EXTERNAL TABLE q18_lm5 (
    intercept            DOUBLE,
    slope                DOUBLE
 )
 ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
 STORED AS TEXTFILE LOCATION '${hiveconf:QUERY_TMP_DIR}/output5/'
;

CREATE EXTERNAL TABLE q18_lm6 (
    intercept            DOUBLE,
    slope                DOUBLE
 )
 ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
 STORED AS TEXTFILE LOCATION '${hiveconf:QUERY_TMP_DIR}/output6/'
;

CREATE EXTERNAL TABLE q18_lm7 (
    intercept            DOUBLE,
    slope                DOUBLE
 )
 ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
 STORED AS TEXTFILE LOCATION '${hiveconf:QUERY_TMP_DIR}/output7/'
;

CREATE EXTERNAL TABLE q18_lm8 (
    intercept            DOUBLE,
    slope                DOUBLE
 )
 ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
 STORED AS TEXTFILE LOCATION '${hiveconf:QUERY_TMP_DIR}/output8/'
;

CREATE EXTERNAL TABLE q18_lm9 (
    intercept            DOUBLE,
    slope                DOUBLE
 )
 ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
 STORED AS TEXTFILE LOCATION '${hiveconf:QUERY_TMP_DIR}/output9/'
;

CREATE EXTERNAL TABLE q18_lm10 (
    intercept            DOUBLE,
    slope                DOUBLE
 )
 ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
 STORED AS TEXTFILE LOCATION '${hiveconf:QUERY_TMP_DIR}/output10/'
;

CREATE EXTERNAL TABLE q18_lm11 (
    intercept            DOUBLE,
    slope                DOUBLE
 )
 ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
 STORED AS TEXTFILE LOCATION '${hiveconf:QUERY_TMP_DIR}/output11/'
;

CREATE EXTERNAL TABLE  q18_lm12 (
    intercept            DOUBLE,
    slope                DOUBLE
 )
 ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
 STORED AS TEXTFILE LOCATION '${hiveconf:QUERY_TMP_DIR}/output12/'
;




-- (!) Set Paths --------------------------------------------------------------
--LOAD DATA INPATH '${hiveconf:QUERY_TMP_DIR}/output1/part-r-00000' OVERWRITE INTO TABLE q18_lm1;
--LOAD DATA INPATH '${hiveconf:QUERY_TMP_DIR}/output2/part-r-00000' OVERWRITE INTO TABLE q18_lm2;
--LOAD DATA INPATH '${hiveconf:QUERY_TMP_DIR}/output3/part-r-00000' OVERWRITE INTO TABLE q18_lm3;
--LOAD DATA INPATH '${hiveconf:QUERY_TMP_DIR}/output4/part-r-00000' OVERWRITE INTO TABLE q18_lm4;
--LOAD DATA INPATH '${hiveconf:QUERY_TMP_DIR}/output5/part-r-00000' OVERWRITE INTO TABLE q18_lm5;
--LOAD DATA INPATH '${hiveconf:QUERY_TMP_DIR}/output6/part-r-00000' OVERWRITE INTO TABLE q18_lm6;
--LOAD DATA INPATH '${hiveconf:QUERY_TMP_DIR}/output7/part-r-00000' OVERWRITE INTO TABLE q18_lm7;
--LOAD DATA INPATH '${hiveconf:QUERY_TMP_DIR}/output8/part-r-00000' OVERWRITE INTO TABLE q18_lm8;
--LOAD DATA INPATH '${hiveconf:QUERY_TMP_DIR}/output9/part-r-00000' OVERWRITE INTO TABLE q18_lm9;
--LOAD DATA INPATH '${hiveconf:QUERY_TMP_DIR}/output10/part-r-00000' OVERWRITE INTO TABLE q18_lm10;
--LOAD DATA INPATH '${hiveconf:QUERY_TMP_DIR}/output11/part-r-00000' OVERWRITE INTO TABLE q18_lm11;
--LOAD DATA INPATH '${hiveconf:QUERY_TMP_DIR}/output12/part-r-00000' OVERWRITE INTO TABLE q18_lm12;


-------------------------------------------------------------------------------

--CREATE RESULT TABLE. Store query result externally in output_dir/qXXresult/
DROP TABLE IF EXISTS q18_store_coefficient;

CREATE TABLE q18_store_coefficient
(
    cat             INT,
    intercept	    DOUBLE,
    slope           DOUBLE
);



INSERT INTO TABLE q18_store_coefficient
    SELECT 1,
           intercept,
           slope        
    FROM q18_lm1;

INSERT INTO TABLE q18_store_coefficient
    SELECT 2,
           intercept,
           slope        
    FROM q18_lm2;

INSERT INTO TABLE q18_store_coefficient
    SELECT 3,
           intercept,
           slope        
    FROM q18_lm3;

INSERT INTO TABLE q18_store_coefficient
    SELECT 4,
           intercept,
           slope        
    FROM q18_lm4;

INSERT INTO TABLE q18_store_coefficient
    SELECT 5,
           intercept,
           slope        
    FROM q18_lm5;

INSERT INTO TABLE q18_store_coefficient
    SELECT 6,
           intercept,
           slope        
    FROM q18_lm6;

INSERT INTO TABLE q18_store_coefficient
    SELECT 7,
           intercept,
           slope        
    FROM q18_lm7;

INSERT INTO TABLE q18_store_coefficient
    SELECT 8,
           intercept,
           slope        
    FROM q18_lm8;

INSERT INTO TABLE q18_store_coefficient
    SELECT 9,
           intercept,
           slope        
    FROM q18_lm9;

INSERT INTO TABLE q18_store_coefficient
    SELECT 10,
           intercept,
           slope        
    FROM q18_lm10;

INSERT INTO TABLE q18_store_coefficient
    SELECT 11,
           intercept,
           slope        
    FROM q18_lm11;

INSERT INTO TABLE q18_store_coefficient
    SELECT 12,
           intercept,
           slope        
    FROM q18_lm12;

-------------------------------------------------------------------------------



--Cleanup




--Display result
--SELECT * FROM ${hiveconf:resultTableName};
