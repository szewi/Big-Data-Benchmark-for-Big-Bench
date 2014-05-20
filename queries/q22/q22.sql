-- Global hive options (see: Big-Bench/setEnvVars)
set hive.exec.parallel=${env:BIG_BENCH_hive_exec_parallel};
set hive.exec.parallel.thread.number=${env:BIG_BENCH_hive_exec_parallel_thread_number};
set hive.exec.compress.intermediate=${env:BIG_BENCH_hive_exec_compress_intermediate};
set mapred.map.output.compression.codec=${env:BIG_BENCH_mapred_map_output_compression_codec};
set hive.exec.compress.output=${env:BIG_BENCH_hive_exec_compress_output};
set mapred.output.compression.codec=${env:BIG_BENCH_mapred_output_compression_codec};
set hive.default.fileformat=${env:BIG_BENCH_hive_default_fileformat};
set hive.optimize.mapjoin.mapreduce=${env:BIG_BENCH_hive_optimize_mapjoin_mapreduce};
set hive.optimize.bucketmapjoin=${env:BIG_BENCH_hive_optimize_bucketmapjoin};
set hive.optimize.bucketmapjoin.sortedmerge=${env:BIG_BENCH_hive_optimize_bucketmapjoin_sortedmerge};
set hive.auto.convert.join=${env:BIG_BENCH_hive_auto_convert_join};
set hive.auto.convert.sortmerge.join=${env:BIG_BENCH_hive_auto_convert_sortmerge_join};
set hive.auto.convert.sortmerge.join.noconditionaltask=${env:BIG_BENCH_hive_auto_convert_sortmerge_join_noconditionaltask};
set hive.optimize.ppd=${env:BIG_BENCH_hive_optimize_ppd};
set hive.optimize.index.filter=${env:BIG_BENCH_hive_optimize_index_filter};

--display settings
set hive.exec.parallel;
set hive.exec.parallel.thread.number;
set hive.exec.compress.intermediate;
set mapred.map.output.compression.codec;
set hive.exec.compress.output;
set mapred.output.compression.codec;
set hive.default.fileformat;
set hive.optimize.mapjoin.mapreduce;
set hive.optimize.bucketmapjoin;
set hive.optimize.bucketmapjoin.sortedmerge;
set hive.auto.convert.join;
set hive.auto.convert.sortmerge.join;
set hive.auto.convert.sortmerge.join.noconditionaltask;

-- Database
use ${env:BIG_BENCH_HIVE_DATABASE};

-- Resources

-- Result file configuration
set QUERY_NUM=q22;
set resultTableName=${hiveconf:QUERY_NUM}result;
set resultFile=${env:BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${hiveconf:resultTableName};




--DROP TABLE IF EXISTS q22_coalition_22;
--CREATE TABLE q22_coalition_22 AS
--  SELECT *
--    FROM inventory inv
--    JOIN (SELECT * 
--            FROM item i 
--           WHERE i.i_current_price > 0.98 
--             AND i.i_current_price < 1.5) i
--         ON inv.inv_item_sk = i.i_item_sk
--    JOIN warehouse w ON inv.inv_warehouse_sk = w.w_warehouse_sk
--    JOIN date_dim d ON inv.inv_date_sk = d.d_date_sk
--   WHERE datediff(d_date, '2000-05-08') >= -30 
--    AND datediff(d_date, '2000-05-08') <= 30;


DROP TABLE IF EXISTS q22_inner;
CREATE TABLE q22_inner AS
SELECT 	w_warehouse_name, 
	i_item_id,
	 sum(CASE WHEN datediff(d_date, '2000-05-08') < 0 
	          THEN inv_quantity_on_hand 
	          ELSE 0 END) AS inv_before,
	 sum(CASE WHEN datediff(d_date, '2000-05-08') >= 0 
	          THEN inv_quantity_on_hand
	          ELSE 0 END) AS inv_after
FROM (
	SELECT *
	    FROM inventory inv
	    JOIN (SELECT * 
		    FROM item i 
		   WHERE i.i_current_price > 0.98 
		     AND i.i_current_price < 1.5
		) i
		 ON inv.inv_item_sk = i.i_item_sk
	    JOIN warehouse w ON inv.inv_warehouse_sk = w.w_warehouse_sk
	    JOIN date_dim d ON inv.inv_date_sk = d.d_date_sk
	   WHERE datediff(d_date, '2000-05-08') >= -30 
	     AND datediff(d_date, '2000-05-08') <= 30

)q22_coalition_22 
GROUP BY w_warehouse_name, i_item_id;




DROP TABLE IF EXISTS q22_conditional_ratio;
CREATE TABLE q22_conditional_ratio AS
  SELECT w_warehouse_name, inv_after/inv_before
    FROM q22_inner
   WHERE inv_before > 0
     AND inv_after/inv_before >= 2.0/3.0 
     AND inv_after/inv_before <= 3.0/2.0;





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
  SELECT name.w_warehouse_name, i_item_id, inv_before, inv_after
    FROM q22_inner name 
    JOIN q22_conditional_ratio nombre 
      ON name.w_warehouse_name = nombre.w_warehouse_name
   ORDER BY w_warehouse_name, i_item_id;


---- cleanup ----------------
DROP TABLE IF EXISTS q22_conditional_ratio;
DROP TABLE IF EXISTS q22_inner;
--DROP TABLE IF EXISTS q22_coalition_22;

