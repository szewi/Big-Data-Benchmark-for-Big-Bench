--Identify the stores with flat or declining sales in 3 consecutive months,
--check if there are any negative reviews regarding these stores available online.

-- Resources



-- Result file configuration
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
STORED AS TEXTFILE LOCATION '${hiveconf:TEMP_DIR}/q18_matrix1'
;

CREATE TABLE q18_matrix2 (d BIGINT, sales BIGINT)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:TEMP_DIR}/q18_matrix2'
;

CREATE TABLE q18_matrix3 (d BIGINT, sales BIGINT)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:TEMP_DIR}/q18_matrix3'
;

CREATE TABLE q18_matrix4 (d BIGINT, sales BIGINT)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:TEMP_DIR}/q18_matrix4'
;

CREATE TABLE q18_matrix5 (d BIGINT, sales BIGINT)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:TEMP_DIR}/q18_matrix5'
;

CREATE TABLE q18_matrix6 (d BIGINT, sales BIGINT)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:TEMP_DIR}/q18_matrix6'
;

CREATE TABLE q18_matrix7 (d BIGINT, sales BIGINT)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:TEMP_DIR}/q18_matrix7'
;

CREATE TABLE q18_matrix8 (d BIGINT, sales BIGINT)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:TEMP_DIR}/q18_matrix8'
;

CREATE TABLE q18_matrix9 (d BIGINT, sales BIGINT)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:TEMP_DIR}/q18_matrix9'
;

CREATE TABLE q18_matrix10 (d BIGINT, sales BIGINT)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:TEMP_DIR}/q18_matrix10'
;

CREATE TABLE q18_matrix11 (d BIGINT, sales BIGINT)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:TEMP_DIR}/q18_matrix11'
;

CREATE TABLE q18_matrix12 (d BIGINT, sales BIGINT)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' 
STORED AS TEXTFILE LOCATION '${hiveconf:TEMP_DIR}/q18_matrix12'
;
	


FROM (
  SELECT 
    ss_store_sk 	 AS store,
    ss_sold_date_sk  AS d,
    sum(ss_net_paid) AS sales
  FROM store_sales s
  --select date range
  LEFT SEMI JOIN (	
			SELECT d_date_sk 
			FROM  date_dim d
			WHERE d.d_date >= '${hiveconf:q18_startDate}'
			AND   d.d_date <= '${hiveconf:q18_endDate}'
		) dd ON ( s.ss_sold_date_sk=dd.d_date_sk ) 
  GROUP BY ss_store_sk, ss_sold_date_sk
) tmp
INSERT OVERWRITE TABLE q18_matrix1  SELECT d, sales WHERE store = 1
INSERT OVERWRITE TABLE q18_matrix2  SELECT d, sales WHERE store = 2
INSERT OVERWRITE TABLE q18_matrix3  SELECT d, sales WHERE store = 3
INSERT OVERWRITE TABLE q18_matrix4  SELECT d, sales WHERE store = 4
INSERT OVERWRITE TABLE q18_matrix5  SELECT d, sales WHERE store = 5
INSERT OVERWRITE TABLE q18_matrix6  SELECT d, sales WHERE store = 6
INSERT OVERWRITE TABLE q18_matrix7  SELECT d, sales WHERE store = 7
INSERT OVERWRITE TABLE q18_matrix8  SELECT d, sales WHERE store = 8
INSERT OVERWRITE TABLE q18_matrix9  SELECT d, sales WHERE store = 9
INSERT OVERWRITE TABLE q18_matrix10 SELECT d, sales WHERE store = 10
INSERT OVERWRITE TABLE q18_matrix11 SELECT d, sales WHERE store = 11
INSERT OVERWRITE TABLE q18_matrix12 SELECT d, sales WHERE store = 12
;

