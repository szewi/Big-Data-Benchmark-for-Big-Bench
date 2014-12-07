
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
    intercept     DOUBLE,
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
