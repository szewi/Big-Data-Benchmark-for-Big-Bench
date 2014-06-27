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

--Find all customers who viewed items of a given category on the web
--in a given month and year that was followed by an in-store purchase in the three
--consecutive months.

--1)
DROP VIEW IF EXISTS q12_click;
CREATE VIEW IF NOT EXISTS q12_click AS
 SELECT c.wcs_item_sk 		AS item,
        c.wcs_user_sk 		AS uid,
        c.wcs_click_date_sk 	AS c_date,
        c.wcs_click_time_sk 	AS c_time
  FROM web_clickstreams c JOIN item i ON c.wcs_item_sk = i.i_item_sk
  WHERE (i.i_category = 'Books' OR i.i_category = 'Electronics')
    AND c.wcs_user_sk is not null
    AND c.wcs_click_date_sk > 36403
    AND c.wcs_click_date_sk < 36403 + 30
  ORDER BY c_date, c_time;

--2)
DROP VIEW IF EXISTS q12_sale;
CREATE VIEW IF NOT EXISTS q12_sale AS 
 SELECT ss.ss_item_sk 		AS item,
        ss.ss_customer_sk 	AS uid,
        ss.ss_sold_date_sk 	AS s_date,
        ss.ss_sold_time_sk 	AS s_time
   FROM store_sales ss JOIN item i ON ss.ss_item_sk = i.i_item_sk
  WHERE (i.i_category = 'Books' OR i.i_category = 'Electronics')
    AND ss.ss_customer_sk is not null
    AND ss.ss_sold_date_sk > 36403
    AND ss.ss_sold_date_sk < 36403 + 120
  ORDER BY s_date, s_time;

--Result  --------------------------------------------------------------------		
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;	

DROP TABLE IF EXISTS ${hiveconf:RESULT_TABLE};
CREATE TABLE ${hiveconf:RESULT_TABLE}
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'
STORED AS ${env:BIG_BENCH_hive_default_fileformat_result_table}  LOCATION '${hiveconf:RESULT_DIR}' 
AS
-- the real query part	
SELECT c_date, s_date, s.uid
  FROM q12_click c JOIN q12_sale s
    ON c.uid = s.uid
 WHERE c.c_date < s.s_date;

--TODO: have to fix partition

------------------------ REGEX CODE GOES HERE ---------------------------------
--TODO check this: Npath seams to be unnecssary. so this query should be done
-------------------------------------------------------------------------------

-- cleanup -------------------------------------------------------------
DROP VIEW IF EXISTS q12_click;
DROP VIEW IF EXISTS q12_sale;
