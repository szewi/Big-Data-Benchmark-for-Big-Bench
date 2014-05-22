-- Global hive options (see: Big-Bench/setEnvVars)
--set hive.exec.parallel=${env:BIG_BENCH_hive_exec_parallel};
--set hive.exec.parallel.thread.number=${env:BIG_BENCH_hive_exec_parallel_thread_number};
--set hive.exec.compress.intermediate=${env:BIG_BENCH_hive_exec_compress_intermediate};
--set mapred.map.output.compression.codec=${env:BIG_BENCH_mapred_map_output_compression_codec};
--set hive.exec.compress.output=${env:BIG_BENCH_hive_exec_compress_output};
--set mapred.output.compression.codec=${env:BIG_BENCH_mapred_output_compression_codec};
--set hive.default.fileformat=${env:BIG_BENCH_hive_default_fileformat};
--set hive.optimize.mapjoin.mapreduce=${env:BIG_BENCH_hive_optimize_mapjoin_mapreduce};
--set hive.optimize.bucketmapjoin=${env:BIG_BENCH_hive_optimize_bucketmapjoin};
--set hive.optimize.bucketmapjoin.sortedmerge=${env:BIG_BENCH_hive_optimize_bucketmapjoin_sortedmerge};
--set hive.auto.convert.join=${env:BIG_BENCH_hive_auto_convert_join};
--set hive.auto.convert.sortmerge.join=${env:BIG_BENCH_hive_auto_convert_sortmerge_join};
--set hive.auto.convert.sortmerge.join.noconditionaltask=${env:BIG_BENCH_hive_auto_convert_sortmerge_join_noconditionaltask};
--set hive.optimize.ppd=${env:BIG_BENCH_hive_optimize_ppd};
--set hive.optimize.index.filter=${env:BIG_BENCH_hive_optimize_index_filter};

--display settings
set hive.exec.parallel;
set hive.exec.parallel.thread.number;
set hive.exec.compress.intermediate;
set mapred.map.output.compression.codec;
set hive.exec.compress.output;
set mapred.output.compression.codec;
set hive.default.fileformat;
set hive.optimize.mapjoin.mapreduce;
set hive.mapjoin.smalltable.filesize;
set hive.optimize.bucketmapjoin;
set hive.optimize.bucketmapjoin.sortedmerge;
set hive.auto.convert.join;
set hive.auto.convert.sortmerge.join;
set hive.auto.convert.sortmerge.join.noconditionaltask;
set hive.optimize.ppd;
set hive.optimize.index.filter;

-- Database
use ${env:BIG_BENCH_HIVE_DATABASE};

-- Resources

-- Result file configuration
set QUERY_NUM=q21;
set resultTableName=${hiveconf:QUERY_NUM}result;
set resultFile=${env:BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${hiveconf:resultTableName};

-- TODO Empty result - needs more testing


--Result --------------------------------------------------------------------		
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;	
--CREATE RESULT TABLE. Store query result externally in output_dir/qXXresult/
DROP TABLE IF EXISTS ${hiveconf:resultTableName};
CREATE TABLE ${hiveconf:resultTableName}
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'
STORED AS ${env:BIG_BENCH_hive_default_fileformat_result_table} LOCATION '${hiveconf:resultFile}' 
AS
-- the real query part
	SELECT 	  i.i_item_id 		AS item_id
		, i.i_item_desc 	AS item_desc
		, s.s_store_id 		AS store_id
		, s.s_store_name 	AS store_name
		, SUM(ss.ss_quantity) 	AS store_sales_quantity
		, SUM(sr.sr_return_quantity) AS store_returns_quantity 
		, SUM(ws.ws_quantity) 	AS web_sales_quantity
	FROM store_sales ss

	JOIN (SELECT d_date_sk FROM date_dim WHERE d_year=1998 and d_moy=4) d1 	
				ON d1.d_date_sk = ss.ss_sold_date_sk  

	JOIN store_returns sr 	ON  sr.sr_customer_sk = ss.ss_customer_sk
				AND ss.ss_item_sk = sr.sr_item_sk
				AND ss.ss_ticket_number = sr.sr_ticket_number

	JOIN (SELECT d_date_sk FROM date_dim  WHERE d_year = 1998 AND d_moy > 4-1 AND d_moy < 4+3+1 ) d2 
				ON  d2.d_date_sk = sr.sr_returned_date_sk 

	JOIN web_sales ws 	ON  sr.sr_item_sk = ws.ws_item_sk
				AND sr.sr_customer_sk = ws.ws_bill_customer_sk

	JOIN (SELECT d_date_sk FROM date_dim  WHERE  d_year in (1998 ,1998+1 ,1998+2) ) d3 
				ON d3.d_date_sk  = ws.ws_sold_date_sk 

	JOIN item i 		ON i.i_item_sk = ss.ss_item_sk
	JOIN store s 		ON s.s_store_sk = ss.ss_store_sk
	GROUP BY i.i_item_id, i.i_item_desc, s.s_store_id, s.s_store_name
	ORDER BY item_id, item_desc, store_id, store_name;


