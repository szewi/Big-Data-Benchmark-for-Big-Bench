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

--Cleanup
DROP TABLE IF EXISTS q18_storeSentiments;
DROP VIEW IF EXISTS q18_store_coefficient;
DROP VIEW IF EXISTS q18_time_series_store;


DROP TABLE IF EXISTS q18_matrix1;
DROP TABLE IF EXISTS q18_matrix2;
DROP TABLE IF EXISTS q18_matrix3;
DROP TABLE IF EXISTS q18_matrix4;
DROP TABLE IF EXISTS q18_matrix5;
DROP TABLE IF EXISTS q18_matrix6;
DROP TABLE IF EXISTS q18_matrix7;
DROP TABLE IF EXISTS q18_matrix8;
DROP TABLE IF EXISTS q18_matrix9;
DROP TABLE IF EXISTS q18_matrix10;
DROP TABLE IF EXISTS q18_matrix11;
DROP TABLE IF EXISTS q18_matrix12;

DROP TABLE IF EXISTS q18_lm1;
DROP TABLE IF EXISTS q18_lm2;
DROP TABLE IF EXISTS q18_lm3;
DROP TABLE IF EXISTS q18_lm4;
DROP TABLE IF EXISTS q18_lm5;
DROP TABLE IF EXISTS q18_lm6;
DROP TABLE IF EXISTS q18_lm7;
DROP TABLE IF EXISTS q18_lm8;
DROP TABLE IF EXISTS q18_lm9;
DROP TABLE IF EXISTS q18_lm10;
DROP TABLE IF EXISTS q18_lm11;
DROP TABLE IF EXISTS q18_lm12;
