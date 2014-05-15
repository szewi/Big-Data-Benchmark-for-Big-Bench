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
set QUERY_NUM=q14;
set resultTableName=${hiveconf:QUERY_NUM}result;
set resultFile=${env:BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${hiveconf:resultTableName};


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
-- Beginn: the real query part
SELECT CAST(amc as double) / CAST(pmc as double) am_pm_ratio
  FROM (SELECT COUNT(*) amc
          FROM web_sales ws
          JOIN household_demographics hd ON ws.ws_ship_hdemo_sk = hd.hd_demo_sk 
           AND hd.hd_dep_count = 5
          JOIN time_dim td ON ws.ws_sold_time_sk = td.t_time_sk 
           AND td.t_hour > 8-1 
           AND td.t_hour < 8+1+1
          JOIN web_page wp ON ws.ws_web_page_sk = wp.wp_web_page_sk 
           AND wp.wp_char_count > 5000-1 
           AND wp.wp_char_count < 5200+1
       ) at JOIN (
        SELECT COUNT(*) pmc
          FROM web_sales ws
          JOIN household_demographics hd ON ws.ws_ship_hdemo_sk = hd.hd_demo_sk 
           AND hd.hd_dep_count = 5
          JOIN time_dim td ON ws.ws_sold_time_sk = td.t_time_sk 
           AND td.t_hour > 19-1 
           AND td.t_hour < 19+1+1
          JOIN web_page wp ON ws.ws_web_page_sk = wp.wp_web_page_sk 
           AND wp.wp_char_count > 5000-1 
           AND wp.wp_char_count < 5200+1
       ) pt
 ORDER BY am_pm_ratio;

