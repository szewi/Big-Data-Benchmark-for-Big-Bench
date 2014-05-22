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
set QUERY_NUM=q24;
set resultTableName=${hiveconf:QUERY_NUM}result;
set resultFile=${env:BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${hiveconf:resultTableName};


-- compute the price change % for the competitor 

DROP VIEW IF EXISTS q24_competitor_price_view;
CREATE VIEW q24_competitor_price_view AS 
  SELECT i_item_sk, (imp_competitor_price - i_current_price)/i_current_price AS price_change,
         imp_start_date, (imp_end_date - imp_start_date) AS no_days
    FROM item i JOIN item_marketprices imp ON i.i_item_sk = imp.imp_item_sk
   WHERE (i.i_item_sk >= 7 AND i.i_item_sk <= 17)
     AND imp.imp_competitor_price < i.i_current_price;



DROP VIEW IF EXISTS q24_self_ws_view;
CREATE VIEW q24_self_ws_view AS 
  SELECT ws_item_sk, 
         SUM(CASE WHEN ws_sold_date_sk >= c.imp_start_date
                   AND ws_sold_date_sk < c.imp_start_date + c.no_days
                  THEN ws_quantity 
                  ELSE 0 END) AS current_ws,
         SUM(CASE WHEN ws_sold_date_sk >= c.imp_start_date - c.no_days 
                   AND ws_sold_date_sk < c.imp_start_date
                  THEN ws_quantity 
                  ELSE 0 END) AS prev_ws
    FROM web_sales ws JOIN q24_competitor_price_view c ON ws.ws_item_sk = c.i_item_sk
   GROUP BY ws_item_sk;



DROP VIEW IF EXISTS q24_self_ss_view;
CREATE VIEW q24_self_ss_view AS 
  SELECT ss_item_sk,
         SUM(CASE WHEN ss_sold_date_sk >= c.imp_start_date
                   AND ss_sold_date_sk < c.imp_start_date + c.no_days 
                  THEN ss_quantity 
                  ELSE 0 END) AS current_ss,
         SUM(CASE WHEN ss_sold_date_sk >= c.imp_start_date - c.no_days 
                   AND ss_sold_date_sk < c.imp_start_date
                  THEN ss_quantity 
                  ELSE 0 END) AS prev_ss
    FROM store_sales ss JOIN q24_competitor_price_view c ON c.i_item_sk = ss.ss_item_sk
   GROUP BY ss_item_sk;


--Result  --------------------------------------------------------------------		
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;	

--CREATE RESULT TABLE. Store query result externally in output_dir/qXXresult/
DROP TABLE IF EXISTS ${hiveconf:resultTableName};
CREATE TABLE ${hiveconf:resultTableName}
ROW FORMAT
DELIMITED FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
STORED AS ${env:BIG_BENCH_hive_default_fileformat_result_table}
LOCATION '${hiveconf:resultFile}' 
AS
-- Begin: the real query part
SELECT i_item_sk, (current_ss + current_ws - prev_ss - prev_ws) / ((prev_ss + prev_ws) * price_change) AS cross_price_elasticity
  FROM q24_competitor_price_view c JOIN q24_self_ws_view ws ON c.i_item_sk = ws.ws_item_sk
  JOIN q24_self_ss_view ss ON c.i_item_sk = ss.ss_item_sk;


-- clean up -----------------------------------
DROP VIEW q24_self_ws_view;
DROP VIEW q24_self_ss_view;
DROP VIEW q24_competitor_price_view;
