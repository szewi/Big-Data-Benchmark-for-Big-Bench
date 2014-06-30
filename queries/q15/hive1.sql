--Find the categories with flat or declining sales for in store purchases
--during a given year for a given store.

-- Resources



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
	SELECT  
		i.i_category_id 	AS cat, -- ranges from 1 to 10
		s.ss_sold_date_sk 	AS d,
		sum(s.ss_net_paid) 	AS sales
	FROM    store_sales s
	-- select date range 
	LEFT SEMI JOIN (	
			SELECT d_date_sk 
			FROM  date_dim d
			WHERE d.d_date >= '${hiveconf:q15_startDate}'
			AND   d.d_date <= '${hiveconf:q15_endDate}'
		) dd ON ( s.ss_sold_date_sk=dd.d_date_sk ) 
	INNER JOIN item i ON s.ss_item_sk = i.i_item_sk 
	WHERE i.i_category_id IS NOT NULL
	  AND s.ss_store_sk = ${hiveconf:q15_store_sk} -- for a given store ranges from 1 to 12
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
