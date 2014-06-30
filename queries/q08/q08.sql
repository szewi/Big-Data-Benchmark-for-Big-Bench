--For online sales, compare the total sales in which customers checked
--online reviews before making the purchase and that of sales in which customers
--did not read reviews. Consider only online sales for a specific category in a given
--year.

-- Resources
ADD FILE ${env:BIG_BENCH_QUERIES_DIR}/q08/q8_reducer.py;


CREATE VIEW IF NOT EXISTS q08_DateRange
AS
	SELECT d_date_sk 
	FROM  date_dim d
	WHERE d.d_date >= '${hiveconf:q08_startDate}'
	AND   d.d_date <= '${hiveconf:q08_endDate}'
;
--!echo "created q08_DateRange";

--PART 1 - sales that users have viewed the review pages--------------------------------------------------------
DROP VIEW if EXISTS q08_tmp_sales_review;
CREATE VIEW if not exists q08_tmp_sales_review AS 
SELECT DISTINCT s_sk 
FROM (

		FROM (	
			SELECT 	
				c.wcs_user_sk 		AS uid, 
				c.wcs_click_date_sk AS c_date, 
				c.wcs_click_time_sk AS c_time, 
				c.wcs_sales_sk 	AS sales_sk, 
				w.wp_type 		AS wpt
			FROM web_clickstreams c 
			JOIN  q08_DateRange d	ON	( c.wcs_click_date_sk = d.d_date_sk)
			INNER JOIN web_page w 	ON 	( c.wcs_web_page_sk = w.wp_web_page_sk 
									  AND c.wcs_user_sk is not null
									)
			CLUSTER BY uid
		) q08_map_output
	REDUCE q08_map_output.uid, 
		q08_map_output.c_date, 
		q08_map_output.c_time, 
		q08_map_output.sales_sk, 
		q08_map_output.wpt
	USING 'python q8_reducer.py ${hiveconf:q08_category}'
	AS (s_date BIGINT, s_sk BIGINT)
) q08npath
;



--PART 2 - helper table: sales within one year starting 1999-09-02  ---------------------------------------
DROP VIEW if EXISTS q08_tmp_webSales_date;
CREATE VIEW IF NOT EXISTS q08_tmp_webSales_date AS 
	SELECT ws_net_paid, ws_order_number
	FROM web_sales ws
	JOIN  q08_DateRange d ON ( ws.ws_sold_date_sk = d.d_date_sk)
;


--PART 3 - for sales in given year, compute sales in which customers checked online reviews vs. sales in which customers did not read reviews.
--Result  --------------------------------------------------------------------		
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;	
--CREATE RESULT TABLE. Store query result externally in output_dir/qXXresult/
DROP TABLE IF EXISTS ${hiveconf:RESULT_TABLE};
CREATE TABLE ${hiveconf:RESULT_TABLE}
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'
STORED AS ${env:BIG_BENCH_hive_default_fileformat_result_table} LOCATION '${hiveconf:RESULT_DIR}' 
AS

-- the real query part----------------------------------------------------------------------
SELECT 	q08_review_sales.amount 			AS q08_review_sales_amount,
	q08_all_sales.amount - q08_review_sales.amount  AS no_q08_review_sales_amount
FROM	(
	SELECT 1 AS id, sum (ws_net_paid) as amount 
	FROM q08_tmp_webSales_date ws
	INNER JOIN q08_tmp_sales_review sr ON ws.ws_order_number = sr.s_sk
) q08_review_sales

JOIN (
	SELECT 1 AS id, sum (ws_net_paid) as amount
	FROM q08_tmp_webSales_date ws
	
)  q08_all_sales 
ON q08_review_sales.id =  q08_all_sales.id
;



--cleanup-------------------------------------------------------------------
DROP VIEW if EXISTS q08_tmp_sales_review;
DROP VIEW if EXISTS q08_tmp_webSales_date;
DROP VIEW IF EXISTS q08_DateRange;
