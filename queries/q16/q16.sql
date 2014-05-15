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

-- Result file configuration
set QUERY_NUM=q16;
set resultTableName=${hiveconf:QUERY_NUM}result;
set resultFile=${env:BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${hiveconf:resultTableName};

--TODO More testing needed



--CREATE RESULT TABLE. Store query result externaly in output_dir/qXXresult/
--Result  --------------------------------------------------------------------		
--kepp result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;	
DROP TABLE IF EXISTS ${hiveconf:resultTableName};
CREATE TABLE ${hiveconf:resultTableName}
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'
STORED AS TEXTFILE LOCATION '${hiveconf:resultFile}' 
AS
-- the real query part
SELECT w_state, i_item_id,
       SUM(CASE 
           WHEN (unix_timestamp(d_date,'yyyy-MM-dd') < unix_timestamp('1998-03-16','yyyy-MM-dd')) 
           THEN ws_sales_price - COALESCE(wr_refunded_cash,0) 
           ELSE 0.0 END) as sales_before,
       SUM(CASE 
           WHEN (unix_timestamp(d_date,'yyyy-MM-dd') >= unix_timestamp('1998-03-16','yyyy-MM-dd')) 
           THEN ws_sales_price - coalesce(wr_refunded_cash,0) 
           ELSE 0.0 END) as sales_after
  FROM (
	SELECT * 
	FROM web_sales ws LEFT OUTER JOIN web_returns wr ON
			(ws.ws_order_number = wr.wr_order_number
			AND ws.ws_item_sk = wr.wr_item_sk)
  ) a1
  JOIN item i ON a1.ws_item_sk = i.i_item_sk
  JOIN warehouse w ON a1.ws_warehouse_sk = w.w_warehouse_sk
  JOIN date_dim d ON a1.ws_sold_date_sk = d.d_date_sk
   AND unix_timestamp(d.d_date, 'yyyy-MM-dd') >= unix_timestamp('1998-02-16', 'yyyy-MM-dd') 
   AND unix_timestamp(d.d_date, 'yyyy-MM-dd') <= unix_timestamp('1998-04-16', 'yyyy-MM-dd')
 GROUP BY w_state,i_item_id
 ORDER BY w_state,i_item_id;

-- cleaning up ---------------------------------------------------------------------

