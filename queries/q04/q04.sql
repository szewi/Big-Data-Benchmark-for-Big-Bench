--Shopping cart abandonment analysis: For users who added products in
--their shopping carts but did not check out in the online store, find the average
--number of pages they visited during their sessions.

-- Resources
ADD FILE ${env:BIG_BENCH_QUERIES_DIR}/q04/q4_mapper1.py;
ADD FILE ${env:BIG_BENCH_QUERIES_DIR}/q04/q4_mapper2.py;
ADD FILE ${env:BIG_BENCH_QUERIES_DIR}/q04/q4_reducer1.py;
ADD FILE ${env:BIG_BENCH_QUERIES_DIR}/q04/q4_reducer2.py;

-- Query parameters

-- Part 1: join webclickstreams with user, webpage and date -----------			  
DROP VIEW IF EXISTS q04_tmp_sessions;
CREATE VIEW q04_tmp_sessions AS 
SELECT * 
FROM (
	FROM (
		SELECT 	
			c.wcs_user_sk 		AS uid , 
			c.wcs_item_sk 		AS item , 
			w.wp_type 		AS wptype , 
			t.t_time+unix_timestamp(d.d_date,'yyyy-MM-dd') AS tstamp
               FROM web_clickstreams c 
               JOIN web_page w ON (c.wcs_web_page_sk = w.wp_web_page_sk 
               				 AND c.wcs_user_sk IS NOT NULL)
               JOIN date_dim d ON c.wcs_click_date_sk = d.d_date_sk
               JOIN time_dim t ON c.wcs_click_time_sk = t.t_time_sk
		CLUSTER BY uid
		) q04_tmp_map_output 
		REDUCE 	  
			  q04_tmp_map_output.uid
			, q04_tmp_map_output.item
			, q04_tmp_map_output.wptype
			, q04_tmp_map_output.tstamp
		USING 'python q4_reducer1.py' AS (
					  uid 	BIGINT
					, item 	BIGINT
					, wptype STRING
					, tstamp BIGINT
					, sessionid STRING)
) q04_tmp_sessionize
--ORDER BY uid, tstamp --ORDER BY is bad! total ordering ->only one reducer
--LIMIT 2500
CLUSTER BY sessionid,uid, tstamp
;


-- Part 2: Abandoned shopping carts ----------------------------------
DROP VIEW IF EXISTS q04_tmp_cart_abandon;
CREATE VIEW q04_tmp_cart_abandon AS 
SELECT * 
FROM (
	FROM q04_tmp_sessions q04_tmp_map_output
	REDUCE 	q04_tmp_map_output.uid, 
		q04_tmp_map_output.item, 
		q04_tmp_map_output.wptype, 
		q04_tmp_map_output.tstamp, 
		q04_tmp_map_output.sessionid
 	USING 'python q4_reducer2.py' AS (sid STRING, start_s BIGINT, end_s BIGINT)
) q04_tmp_npath
CLUSTER BY sid
;

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
-- the real query part
SELECT c.sid, COUNT (*) AS s_pages
FROM q04_tmp_cart_abandon c 
JOIN q04_tmp_sessions s ON s.sessionid = c.sid
GROUP BY c.sid
;


--cleanup --------------------------------------------
DROP VIEW IF EXISTS q04_tmp_sessions;
DROP VIEW IF EXISTS q04_tmp_cart_abandon;
