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

DROP TABLE IF EXISTS ${hiveconf:LM_BASENAME}1;
DROP TABLE IF EXISTS ${hiveconf:LM_BASENAME}2;
DROP TABLE IF EXISTS ${hiveconf:LM_BASENAME}3;
DROP TABLE IF EXISTS ${hiveconf:LM_BASENAME}4;
DROP TABLE IF EXISTS ${hiveconf:LM_BASENAME}5;
DROP TABLE IF EXISTS ${hiveconf:LM_BASENAME}6;
DROP TABLE IF EXISTS ${hiveconf:LM_BASENAME}7;
DROP TABLE IF EXISTS ${hiveconf:LM_BASENAME}8;
DROP TABLE IF EXISTS ${hiveconf:LM_BASENAME}9;
DROP TABLE IF EXISTS ${hiveconf:LM_BASENAME}10;


CREATE EXTERNAL TABLE ${hiveconf:LM_BASENAME}1 (
    intercept            DOUBLE,
    slope                DOUBLE
 )
 ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
 STORED AS TEXTFILE LOCATION '${hiveconf:LM_BASEDIR}1'
;

CREATE EXTERNAL TABLE ${hiveconf:LM_BASENAME}2 (
    intercept            DOUBLE,
    slope                DOUBLE
 )
 ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
 STORED AS TEXTFILE LOCATION '${hiveconf:LM_BASEDIR}2'
;

CREATE EXTERNAL TABLE ${hiveconf:LM_BASENAME}3 (
    intercept            DOUBLE,
    slope                DOUBLE
 )
 ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
 STORED AS TEXTFILE LOCATION '${hiveconf:LM_BASEDIR}3'
;

CREATE EXTERNAL TABLE ${hiveconf:LM_BASENAME}4 (
    intercept            DOUBLE,
    slope                DOUBLE
 )
 ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
 STORED AS TEXTFILE LOCATION '${hiveconf:LM_BASEDIR}4'
;

CREATE EXTERNAL TABLE ${hiveconf:LM_BASENAME}5 (
    intercept            DOUBLE,
    slope                DOUBLE
 )
 ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
 STORED AS TEXTFILE LOCATION '${hiveconf:LM_BASEDIR}5'
;

CREATE EXTERNAL TABLE ${hiveconf:LM_BASENAME}6 (
    intercept            DOUBLE,
    slope                DOUBLE
 )
 ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
 STORED AS TEXTFILE LOCATION '${hiveconf:LM_BASEDIR}6'
;

CREATE EXTERNAL TABLE ${hiveconf:LM_BASENAME}7 (
    intercept            DOUBLE,
    slope                DOUBLE
 )
 ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
 STORED AS TEXTFILE LOCATION '${hiveconf:LM_BASEDIR}7'
;

CREATE EXTERNAL TABLE ${hiveconf:LM_BASENAME}8 (
    intercept            DOUBLE,
    slope                DOUBLE
 )
 ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
 STORED AS TEXTFILE LOCATION '${hiveconf:LM_BASEDIR}8'
;

CREATE EXTERNAL TABLE ${hiveconf:LM_BASENAME}9 (
    intercept            DOUBLE,
    slope                DOUBLE
 )
 ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
 STORED AS TEXTFILE LOCATION '${hiveconf:LM_BASEDIR}9'
;

CREATE EXTERNAL TABLE ${hiveconf:LM_BASENAME}10 (
    intercept            DOUBLE,
    slope                DOUBLE
 )
 ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
 STORED AS TEXTFILE LOCATION '${hiveconf:LM_BASEDIR}10'
;



-- (!) Set Paths --------------------------------------------------------------
--LOAD DATA INPATH '${hiveconf:LM_BASEDIR}1/part-r-00000' OVERWRITE INTO TABLE ${hiveconf:LM_BASENAME}1;
--LOAD DATA INPATH '${hiveconf:LM_BASEDIR}2/part-r-00000' OVERWRITE INTO TABLE ${hiveconf:LM_BASENAME}2;
--LOAD DATA INPATH '${hiveconf:LM_BASEDIR}3/part-r-00000' OVERWRITE INTO TABLE ${hiveconf:LM_BASENAME}3;
--LOAD DATA INPATH '${hiveconf:LM_BASEDIR}4/part-r-00000' OVERWRITE INTO TABLE ${hiveconf:LM_BASENAME}4;
--LOAD DATA INPATH '${hiveconf:LM_BASEDIR}5/part-r-00000' OVERWRITE INTO TABLE ${hiveconf:LM_BASENAME}5;
--LOAD DATA INPATH '${hiveconf:LM_BASEDIR}6/part-r-00000' OVERWRITE INTO TABLE ${hiveconf:LM_BASENAME}6;
--LOAD DATA INPATH '${hiveconf:LM_BASEDIR}7/part-r-00000' OVERWRITE INTO TABLE ${hiveconf:LM_BASENAME}7;
--LOAD DATA INPATH '${hiveconf:LM_BASEDIR}8/part-r-00000' OVERWRITE INTO TABLE ${hiveconf:LM_BASENAME}8;
--LOAD DATA INPATH '${hiveconf:LM_BASEDIR}9/part-r-00000' OVERWRITE INTO TABLE ${hiveconf:LM_BASENAME}9;
--LOAD DATA INPATH '${hiveconf:LM_BASEDIR}10/part-r-00000' OVERWRITE INTO TABLE ${hiveconf:LM_BASENAME}10;


--Result  --------------------------------------------------------------------		
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;
--CREATE RESULT TABLE. Store query result externally in output_dir/qXXresult/
DROP TABLE IF EXISTS ${hiveconf:RESULT_TABLE};

CREATE TABLE ${hiveconf:RESULT_TABLE}
(
    cat             INT,
    intercept	    DOUBLE,
    slope           DOUBLE
)
ROW FORMAT
DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'
STORED AS ${env:BIG_BENCH_hive_default_fileformat_result_table} LOCATION '${hiveconf:RESULT_DIR}' 
;



INSERT INTO TABLE ${hiveconf:RESULT_TABLE}
    SELECT 1,
           intercept,
           slope
    FROM ${hiveconf:LM_BASENAME}1
    WHERE slope < 0;

INSERT INTO TABLE ${hiveconf:RESULT_TABLE}
    SELECT 2,
           intercept,
           slope
    FROM ${hiveconf:LM_BASENAME}2
    WHERE slope < 0;

INSERT INTO TABLE ${hiveconf:RESULT_TABLE}
    SELECT 3,
           intercept,
           slope
    FROM ${hiveconf:LM_BASENAME}3
    WHERE slope < 0;

INSERT INTO TABLE ${hiveconf:RESULT_TABLE}
    SELECT 4,
           intercept,
           slope
    FROM ${hiveconf:LM_BASENAME}4
    WHERE slope < 0;

INSERT INTO TABLE ${hiveconf:RESULT_TABLE}
    SELECT 5,
           intercept,
           slope
    FROM ${hiveconf:LM_BASENAME}5
    WHERE slope < 0;

INSERT INTO TABLE ${hiveconf:RESULT_TABLE}
    SELECT 6,
           intercept,
           slope
    FROM ${hiveconf:LM_BASENAME}6
    WHERE slope < 0;

INSERT INTO TABLE ${hiveconf:RESULT_TABLE}
    SELECT 7,
           intercept,
           slope
    FROM ${hiveconf:LM_BASENAME}7
    WHERE slope < 0;

INSERT INTO TABLE ${hiveconf:RESULT_TABLE}
    SELECT 8,
           intercept,
           slope
    FROM ${hiveconf:LM_BASENAME}8
    WHERE slope < 0;

INSERT INTO TABLE ${hiveconf:RESULT_TABLE}
    SELECT 9,
           intercept,
           slope
    FROM ${hiveconf:LM_BASENAME}9
    WHERE slope < 0;

INSERT INTO TABLE ${hiveconf:RESULT_TABLE}
    SELECT 10,
           intercept,
           slope
    FROM ${hiveconf:LM_BASENAME}10
    WHERE slope < 0;

----cleaning up ----------------------------------------------------------------------

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

DROP TABLE IF EXISTS ${hiveconf:LM_BASENAME}1;
DROP TABLE IF EXISTS ${hiveconf:LM_BASENAME}2;
DROP TABLE IF EXISTS ${hiveconf:LM_BASENAME}3;
DROP TABLE IF EXISTS ${hiveconf:LM_BASENAME}4;
DROP TABLE IF EXISTS ${hiveconf:LM_BASENAME}5;
DROP TABLE IF EXISTS ${hiveconf:LM_BASENAME}6;
DROP TABLE IF EXISTS ${hiveconf:LM_BASENAME}7;
DROP TABLE IF EXISTS ${hiveconf:LM_BASENAME}8;
DROP TABLE IF EXISTS ${hiveconf:LM_BASENAME}9;
DROP TABLE IF EXISTS ${hiveconf:LM_BASENAME}10;
