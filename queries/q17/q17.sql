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
set QUERY_NUM=q17;
set resultTableName=${hiveconf:QUERY_NUM}result;
set resultFile=${env:BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${hiveconf:resultTableName};

--TODO Empty result - needs more testing

--Result  --------------------------------------------------------------------		
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

SELECT promotions, total, promotions/total*100 
--no need to cast promotions/total: sum(COL) return DOUBLE
FROM (
	SELECT SUM(ss_ext_sales_price) promotions
	FROM store_sales ss
	JOIN store s ON ss.ss_store_sk = s.s_store_sk
	JOIN promotion p ON ss.ss_promo_sk = p.p_promo_sk
	JOIN date_dim dd ON ss.ss_sold_date_sk = dd.d_date_sk
	JOIN (
		SELECT * 
		FROM customer c 
		JOIN customer_address ca ON c.c_current_addr_sk = ca.ca_address_sk
	) cc 
	ON ss.ss_customer_sk = cc.c_customer_sk

	JOIN item i ON ss.ss_item_sk = i.i_item_sk
	WHERE ca_gmt_offset = -7
	  AND i_category = 'Jewelry'
	  AND (p_channel_dmail = 'Y' OR p_channel_email = 'Y' OR p_channel_tv = 'Y')
	  AND s_gmt_offset = -7
	  AND d_year = 2001
	  AND d_moy  = 12) promotional_sales
	 
	JOIN (
		SELECT SUM(ss_ext_sales_price) total
		FROM store_sales ss
		JOIN store s ON ss.ss_store_sk = s.s_store_sk
		JOIN date_dim dd ON ss.ss_sold_date_SK = dd.d_date_sk
		JOIN (
			SELECT * 
			FROM customer c 
			JOIN customer_address ca ON c.c_current_addr_sk = ca.ca_address_sk
		) cc
		ON ss.ss_customer_sk = cc.c_customer_sk
         JOIN item i ON ss.ss_item_sk = i.i_item_sk
         WHERE ca_gmt_offset = -7
           AND i_category = 'Jewelry'
           AND s_gmt_offset = -7
           AND d_year = 2001
           AND d_moy  = 12
) all_sales
ORDER BY promotions, total;

