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

------ create input table for mahout --------------------------------------
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;
DROP TABLE IF EXISTS ${hiveconf:TEMP_TABLE};
CREATE TABLE ${hiveconf:TEMP_TABLE} (cid INT
			, id1 INT
			, id3 INT
			, id5 INT
			, id7 INT
			, id9 INT
			, id11 INT
			, id13 INT
			, id15 INT
			, id2 INT
			, id4 INT
			, id6 INT
			, id8 INT
			, id10 INT
			, id14 INT
			, id16 INT) 
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ' '
LINES TERMINATED BY '\n'
STORED AS TEXTFILE
LOCATION '${hiveconf:TEMP_DIR}';

INSERT INTO TABLE ${hiveconf:TEMP_TABLE}
 SELECT ss.ss_customer_sk AS cid,
        count(CASE WHEN i.i_class_id=1  THEN 1 ELSE NULL END) AS id1,
        count(CASE WHEN i.i_class_id=3  THEN 1 ELSE NULL END) AS id3,
        count(CASE WHEN i.i_class_id=5  THEN 1 ELSE NULL END) AS id5,
        count(CASE WHEN i.i_class_id=7  THEN 1 ELSE NULL END) AS id7,
        count(CASE WHEN i.i_class_id=9  THEN 1 ELSE NULL END) AS id9,
        count(CASE WHEN i.i_class_id=11 THEN 1 ELSE NULL END) AS id11,
        count(CASE WHEN i.i_class_id=13 THEN 1 ELSE NULL END) AS id13,
        count(CASE WHEN i.i_class_id=15 THEN 1 ELSE NULL END) AS id15,
        count(CASE WHEN i.i_class_id=2  THEN 1 ELSE NULL END) AS id2,
        count(CASE WHEN i.i_class_id=4  THEN 1 ELSE NULL END) AS id4,
        count(CASE WHEN i.i_class_id=6  THEN 1 ELSE NULL END) AS id6,
        count(CASE WHEN i.i_class_id=8  THEN 1 ELSE NULL END) AS id8,
        count(CASE WHEN i.i_class_id=10 THEN 1 ELSE NULL END) AS id10,
        count(CASE WHEN i.i_class_id=14 THEN 1 ELSE NULL END) AS id14,
        count(CASE WHEN i.i_class_id=16 THEN 1 ELSE NULL END) AS id16
 FROM store_sales ss
      INNER JOIN item i ON ss.ss_item_sk = i.i_item_sk
 WHERE i.i_category = 'Books'
   AND ss.ss_customer_sk IS NOT NULL
 GROUP BY ss.ss_customer_sk
 HAVING count(ss.ss_item_sk) > 5
;

-------------------------------------------------------------------------------
