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
-- Begin: the real query part
SELECT CAST(amc as double) / CAST(pmc as double) am_pm_ratio
FROM (SELECT COUNT(*) amc
          FROM web_sales ws
          JOIN household_demographics hd ON  hd.hd_demo_sk =ws.ws_ship_hdemo_sk
           AND hd.hd_dep_count = 5
          JOIN time_dim td ON td.t_time_sk = ws.ws_sold_time_sk 
           AND td.t_hour > 8-1 
           AND td.t_hour < 8+1+1
          JOIN web_page wp ON  wp.wp_web_page_sk =ws.ws_web_page_sk 
           AND wp.wp_char_count > 5000-1 
           AND wp.wp_char_count < 5200+1
) at 
JOIN (
        SELECT COUNT(*) pmc
          FROM web_sales ws
          JOIN household_demographics hd ON ws.ws_ship_hdemo_sk = hd.hd_demo_sk 
           AND hd.hd_dep_count = 5
          JOIN time_dim td ON  td.t_time_sk =ws.ws_sold_time_sk
           AND td.t_hour > 19-1 
           AND td.t_hour < 19+1+1
          JOIN web_page wp ON wp.wp_web_page_sk = ws.ws_web_page_sk 
           AND wp.wp_char_count > 5000-1 
           AND wp.wp_char_count < 5200+1
) pt
ORDER BY am_pm_ratio;
