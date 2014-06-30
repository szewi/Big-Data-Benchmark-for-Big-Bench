--Find all customers who viewed items of a given category on the web
--in a given month and year that was followed by an in-store purchase in the three
--consecutive months.

-- Resources




--1)
DROP VIEW IF EXISTS q12_click;
CREATE VIEW IF NOT EXISTS q12_click AS
SELECT 	c.wcs_item_sk 			AS item,
		c.wcs_user_sk 			AS uid,
		c.wcs_click_date_sk 	AS c_date,
		c.wcs_click_time_sk 	AS c_time
FROM web_clickstreams c
LEFT SEMI JOIN (	
			SELECT d_date_sk 
			FROM  date_dim d
			WHERE d.d_date >= '${hiveconf:q12_startDate}'
			AND   d.d_date <= '${hiveconf:q12_endDate1}'
		) dd ON ( c.wcs_click_date_sk=dd.d_date_sk ) 
JOIN item i ON c.wcs_item_sk = i.i_item_sk
WHERE i.i_category IN (${hiveconf:q12_i_category_IN})
AND c.wcs_user_sk is not null
CLUSTER BY c_date, c_time
;


--2)
DROP VIEW IF EXISTS q12_sale;
CREATE VIEW IF NOT EXISTS q12_sale AS 
SELECT 	ss.ss_item_sk 		AS item,
		ss.ss_customer_sk 	AS uid,
		ss.ss_sold_date_sk 	AS s_date,
		ss.ss_sold_time_sk 	AS s_time
FROM store_sales ss 
LEFT SEMI JOIN (	
	SELECT d_date_sk 
	FROM  date_dim d
	WHERE d.d_date >= '${hiveconf:q12_startDate}'
	AND   d.d_date <= '${hiveconf:q12_endDate2}'
) dd ON ( ss.ss_sold_date_sk=dd.d_date_sk )
JOIN item i ON ss.ss_item_sk = i.i_item_sk
WHERE i.i_category IN (${hiveconf:q12_i_category_IN})
AND ss.ss_customer_sk is not null
CLUSTER BY s_date, s_time
;


--Result  --------------------------------------------------------------------		
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;	

DROP TABLE IF EXISTS ${hiveconf:RESULT_TABLE};
CREATE TABLE ${hiveconf:RESULT_TABLE}
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'
STORED AS ${env:BIG_BENCH_hive_default_fileformat_result_table}  LOCATION '${hiveconf:RESULT_DIR}' 
AS
-- the real query part	
SELECT c_date, s_date, s.uid
FROM q12_click c 
JOIN q12_sale s ON c.uid = s.uid
WHERE c.c_date < s.s_date;

--TODO: have to fix partition



-- cleanup -------------------------------------------------------------
DROP VIEW IF EXISTS q12_click;
DROP VIEW IF EXISTS q12_sale;
