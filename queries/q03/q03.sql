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
--ADD FILE ${env:BIG_BENCH_QUERIES_DIR}/q03/mapper_q3.py;
ADD FILE ${env:BIG_BENCH_QUERIES_DIR}/q03/reducer_q3.py;

-- Result file configuration
set QUERY_NUM=q03;
set resultTableName=${hiveconf:QUERY_NUM}result;
set resultFile=${env:BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${hiveconf:resultTableName};

--Result -------------------------------------------------------------------------
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
SELECT lastviewed_item, purchased_item, count(*) 
FROM (
        FROM (	
		SELECT 
			wcs_user_sk AS user,
		        wcs_click_date_sk AS lastviewed_date,
		 	wcs_click_time_sk AS lastviewed_time,
			wcs_item_sk AS lastviewed_item,
		        wcs_sales_sk AS lastviewed_sale
		 FROM web_clickstreams
		 CLUSTER BY user
        ) q03_map_output
        REDUCE 
            q03_map_output.user, 
            q03_map_output.lastviewed_date,
	    q03_map_output.lastviewed_time,
            q03_map_output.lastviewed_item,
            q03_map_output.lastviewed_sale
        USING 'python reducer_q3.py'
        AS (lastviewed_item BIGINT, purchased_item BIGINT)
) q03_nPath
WHERE purchased_item = 16891 
GROUP BY lastviewed_item, purchased_item;



