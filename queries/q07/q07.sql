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

DROP TABLE IF EXISTS ${hiveconf:TEMP_TABLE1};
DROP TABLE IF EXISTS ${hiveconf:TEMP_TABLE2};

CREATE TABLE ${hiveconf:TEMP_TABLE1} AS
       SELECT DISTINCT(d_month_seq) AS d_month_seq
         FROM date_dim 
        WHERE d_year = 2002 AND d_moy = 7
;


CREATE TABLE ${hiveconf:TEMP_TABLE2} AS
       SELECT i_category AS i_category, 
              AVG(i_current_price) * 1.2 AS avg_price
         FROM item
        GROUP BY i_category
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
SELECT q07_temp.ca_state AS state, COUNT(*) AS cnt
FROM (
	SELECT    a.ca_state 		AS ca_state
		, i.i_current_price 	AS i_current_price
		, p.avg_price 		AS avg_price 
	FROM customer_address a
               JOIN customer 		  c ON a.ca_address_sk 	= c.c_current_addr_sk
               JOIN store_sales 	  s ON c.c_customer_sk 	= s.ss_customer_sk
               JOIN date_dim 		  d ON s.ss_sold_date_sk = d.d_date_sk
               JOIN item 		  i ON s.ss_item_sk 	= i.i_item_sk
               JOIN ${hiveconf:TEMP_TABLE1} m ON d.d_month_seq 	= m.d_month_seq
               JOIN ${hiveconf:TEMP_TABLE2}  p ON p.i_category 	= i.i_category
) q07_temp 
WHERE q07_temp.i_current_price > q07_temp.avg_price
GROUP BY q07_temp.ca_state
HAVING COUNT(*) >= 10
ORDER BY cnt
LIMIT 100;


--Cleanup ----------------------------------------------------------------
DROP TABLE IF EXISTS ${hiveconf:TEMP_TABLE1};
DROP TABLE IF EXISTS ${hiveconf:TEMP_TABLE2};
