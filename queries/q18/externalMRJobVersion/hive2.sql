-- Resources

-- Result file configuration

-------------------------------------------------------------------------------

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
DROP TABLE IF EXISTS ${hiveconf:LM_BASENAME}11;
DROP TABLE IF EXISTS ${hiveconf:LM_BASENAME}12;

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

CREATE EXTERNAL TABLE ${hiveconf:LM_BASENAME}11 (
    intercept            DOUBLE,
    slope                DOUBLE
 )
 ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
 STORED AS TEXTFILE LOCATION '${hiveconf:LM_BASEDIR}11'
;

CREATE EXTERNAL TABLE  ${hiveconf:LM_BASENAME}12 (
    intercept            DOUBLE,
    slope                DOUBLE
 )
 ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
 STORED AS TEXTFILE LOCATION '${hiveconf:LM_BASEDIR}12'
;




-- (!) Set Paths --------------------------------------------------------------
--LOAD DATA INPATH '${hiveconf:TEMP_DIR}/output1/part-r-00000' OVERWRITE INTO TABLE ${hiveconf:LM_BASENAME}1;
--LOAD DATA INPATH '${hiveconf:TEMP_DIR}/output2/part-r-00000' OVERWRITE INTO TABLE ${hiveconf:LM_BASENAME}2;
--LOAD DATA INPATH '${hiveconf:TEMP_DIR}/output3/part-r-00000' OVERWRITE INTO TABLE ${hiveconf:LM_BASENAME}3;
--LOAD DATA INPATH '${hiveconf:TEMP_DIR}/output4/part-r-00000' OVERWRITE INTO TABLE ${hiveconf:LM_BASENAME}4;
--LOAD DATA INPATH '${hiveconf:TEMP_DIR}/output5/part-r-00000' OVERWRITE INTO TABLE ${hiveconf:LM_BASENAME}5;
--LOAD DATA INPATH '${hiveconf:TEMP_DIR}/output6/part-r-00000' OVERWRITE INTO TABLE ${hiveconf:LM_BASENAME}6;
--LOAD DATA INPATH '${hiveconf:TEMP_DIR}/output7/part-r-00000' OVERWRITE INTO TABLE ${hiveconf:LM_BASENAME}7;
--LOAD DATA INPATH '${hiveconf:TEMP_DIR}/output8/part-r-00000' OVERWRITE INTO TABLE ${hiveconf:LM_BASENAME}8;
--LOAD DATA INPATH '${hiveconf:TEMP_DIR}/output9/part-r-00000' OVERWRITE INTO TABLE ${hiveconf:LM_BASENAME}9;
--LOAD DATA INPATH '${hiveconf:TEMP_DIR}/output10/part-r-00000' OVERWRITE INTO TABLE ${hiveconf:LM_BASENAME}10;
--LOAD DATA INPATH '${hiveconf:TEMP_DIR}/output11/part-r-00000' OVERWRITE INTO TABLE ${hiveconf:LM_BASENAME}11;
--LOAD DATA INPATH '${hiveconf:TEMP_DIR}/output12/part-r-00000' OVERWRITE INTO TABLE ${hiveconf:LM_BASENAME}12;


-------------------------------------------------------------------------------

--CREATE RESULT TABLE. Store query result externally in output_dir/qXXresult/
-- !echo concatenate result files ${hiveconf:LM_BASENAME}1-12 into table: ${hiveconf:TEMP_TABLE};
DROP TABLE IF EXISTS ${hiveconf:TEMP_TABLE};
CREATE TABLE ${hiveconf:TEMP_TABLE}
(
    cat             INT,
    intercept     DOUBLE,
    slope           DOUBLE
);



INSERT INTO TABLE ${hiveconf:TEMP_TABLE}
    SELECT 1,
           intercept,
           slope        
    FROM ${hiveconf:LM_BASENAME}1;

INSERT INTO TABLE ${hiveconf:TEMP_TABLE}
    SELECT 2,
           intercept,
           slope
    FROM ${hiveconf:LM_BASENAME}2;

INSERT INTO TABLE ${hiveconf:TEMP_TABLE}
    SELECT 3,
           intercept,
           slope
    FROM ${hiveconf:LM_BASENAME}3;

INSERT INTO TABLE ${hiveconf:TEMP_TABLE}
    SELECT 4,
           intercept,
           slope
    FROM ${hiveconf:LM_BASENAME}4;

INSERT INTO TABLE ${hiveconf:TEMP_TABLE}
    SELECT 5,
           intercept,
           slope
    FROM ${hiveconf:LM_BASENAME}5;

INSERT INTO TABLE ${hiveconf:TEMP_TABLE}
    SELECT 6,
           intercept,
           slope
    FROM ${hiveconf:LM_BASENAME}6;

INSERT INTO TABLE ${hiveconf:TEMP_TABLE}
    SELECT 7,
           intercept,
           slope
    FROM ${hiveconf:LM_BASENAME}7;

INSERT INTO TABLE ${hiveconf:TEMP_TABLE}
    SELECT 8,
           intercept,
           slope
    FROM ${hiveconf:LM_BASENAME}8;

INSERT INTO TABLE ${hiveconf:TEMP_TABLE}
    SELECT 9,
           intercept,
           slope
    FROM ${hiveconf:LM_BASENAME}9;

INSERT INTO TABLE ${hiveconf:TEMP_TABLE}
    SELECT 10,
           intercept,
           slope
    FROM ${hiveconf:LM_BASENAME}10;

INSERT INTO TABLE ${hiveconf:TEMP_TABLE}
    SELECT 11,
           intercept,
           slope
    FROM ${hiveconf:LM_BASENAME}11;

INSERT INTO TABLE ${hiveconf:TEMP_TABLE}
    SELECT 12,
           intercept,
           slope
    FROM ${hiveconf:LM_BASENAME}12;

-------------------------------------------------------------------------------



--Cleanup




--Display result
--SELECT * FROM ${hiveconf:RESULT_TABLE};
