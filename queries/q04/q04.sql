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
ADD FILE ${env:BIG_BENCH_QUERIES_DIR}/q04/q4_mapper1.py;
ADD FILE ${env:BIG_BENCH_QUERIES_DIR}/q04/q4_mapper2.py;
ADD FILE ${env:BIG_BENCH_QUERIES_DIR}/q04/q4_reducer1.py;
ADD FILE ${env:BIG_BENCH_QUERIES_DIR}/q04/q4_reducer2.py;

-- Result file configuration
set QUERY_NUM=q04;
set resultTableName=${hiveconf:QUERY_NUM}result;
set resultFile=${env:BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${hiveconf:resultTableName};


-- Part 1: join webclickstreams with user, webpage and date -----------			  
DROP VIEW IF EXISTS q04_tmp_sessions;
CREATE VIEW q04_tmp_sessions AS 
SELECT * 
FROM (
	FROM (
		--FROM (
		SELECT 	c.wcs_user_sk AS uid , 
			c.wcs_item_sk AS item , 
			w.wp_type AS wptype , 
			t.t_time+unix_timestamp(d.d_date,'yyyy-MM-dd') AS tstamp
                FROM web_clickstreams c 
                JOIN web_page w ON c.wcs_web_page_sk = w.wp_web_page_sk 
                		AND c.wcs_user_sk IS NOT NULL
                JOIN date_dim d ON c.wcs_click_date_sk = d.d_date_sk
                JOIN time_dim t ON c.wcs_click_time_sk = t.t_time_sk
		--) select_temp
		--MAP select_temp.uid, select_temp.item, select_temp.wptype, select_temp.tstamp
		-- USING 'python q4_mapper1.py' AS uid, item, wptype, tstamp  ||| use cat instead beacause this is a no-op mapper
		--USING 'cat' AS uid, item, wptype, tstamp
		CLUSTER BY uid
	) q04_tmp_map_output 
        REDUCE q04_tmp_map_output.uid, q04_tmp_map_output.item, q04_tmp_map_output.wptype, q04_tmp_map_output.tstamp
        USING 'python q4_reducer1.py' AS (uid BIGINT, item BIGINT, wptype STRING, tstamp BIGINT, sessionid STRING)
) q04_tmp_sessionize
ORDER BY uid, tstamp
CLUSTER BY sessionid
;
--LIMIT 2500;

-- Part 2: Abandoned shopping carts ----------------------------------
DROP VIEW IF EXISTS q04_tmp_cart_abandon;
CREATE VIEW q04_tmp_cart_abandon AS 
SELECT * 
FROM (
	--FROM (
	--	FROM q04_tmp_sessions
        -- 	MAP 	q04_tmp_sessions.uid, 
	--		q04_tmp_sessions.item, 
	--		q04_tmp_sessions.wptype, 
	--		q04_tmp_sessions.tstamp, 
	--		q04_tmp_sessions.sessionid
	--	-- USING 'python q4_mapper2.py'   AS uid, item, wptype, tstamp, sessionid  ||| use cat instead beacause this is a no-op mapper
       	--	USING 'cat'   AS uid, item, wptype, tstamp, sessionid
     	--	CLUSTER BY sessionid
	--) q04_tmp_map_output
	FROM q04_tmp_sessions q04_tmp_map_output
	REDUCE 	q04_tmp_map_output.uid, 
		q04_tmp_map_output.item, 
		q04_tmp_map_output.wptype, 
		q04_tmp_map_output.tstamp, 
		q04_tmp_map_output.sessionid
 	USING 'python q4_reducer2.py'   AS (sid STRING, start_s BIGINT, end_s BIGINT)
) q04_tmp_npath
CLUSTER BY sid
;

--Result  --------------------------------------------------------------------		
--kepp result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;	
--CREATE RESULT TABLE. Store query result externaly in output_dir/qXXresult/
DROP TABLE IF EXISTS ${hiveconf:resultTableName};
CREATE TABLE ${hiveconf:resultTableName}
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'
STORED AS TEXTFILE LOCATION '${hiveconf:resultFile}' 
AS
-- the real query part
SELECT c.sid, COUNT (*) AS s_pages
FROM q04_tmp_cart_abandon c JOIN q04_tmp_sessions s ON s.sessionid = c.sid
GROUP BY c.sid;


--cleanup --------------------------------------------
DROP VIEW IF EXISTS q04_tmp_sessions;
DROP VIEW IF EXISTS q04_tmp_cart_abandon;

