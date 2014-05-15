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
-------------------------------------------------------------------------------    
--Part 1:
-------------------------------------------------------------------------------    
DROP TABLE IF EXISTS q30_c_affinity_input;


CREATE TABLE q30_c_affinity_input AS
  SELECT i.i_category_id AS category_cd,
         s.wcs_user_sk AS customer_id,
         i.i_item_sk AS item
    FROM web_clickstreams s JOIN item i ON s.wcs_item_sk = i.i_item_sk
   WHERE s.wcs_item_sk IS NOT NULL
     AND i.i_category_id IS NOT NULL
     AND s.wcs_user_sk IS NOT NULL;

