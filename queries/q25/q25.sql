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



-- ss_sold_date_sk > 2002-01-02
DROP TABLE IF EXISTS q25_usersegments;
CREATE TABLE q25_usersegments AS
SELECT 
    ss_customer_sk 	AS cid,
    ss_ticket_number 	AS oid,
    ss_sold_date_sk 	AS dateid,
    sum(ss_net_paid) 	AS amount
FROM  store_sales
WHERE ss_sold_date_sk > 37256
  AND ss_customer_sk IS NOT NULL
GROUP BY ss_customer_sk, ss_ticket_number, ss_sold_date_sk
;



INSERT INTO TABLE q25_usersegments
SELECT 
    ws_bill_customer_sk AS cid,
    ws_order_number 	AS oid,
    ws_sold_date_sk 	AS dateid,
    sum(ws_net_paid) 	AS amount
FROM web_sales
WHERE ws_sold_date_sk > 37256
  AND ws_bill_customer_sk IS NOT NULL
GROUP BY ws_bill_customer_sk, ws_order_number, ws_sold_date_sk;

-------------------------------------------------------------------------------
--2003-01-02 == date_sk 37621

------ create input table for mahout --------------------------------------
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;	

DROP TABLE IF EXISTS q25_ctable;
CREATE TABLE q25_ctable (	  cid INT
				, recency INT
				, frequency INT
				, totalspend INT) 
       ROW FORMAT DELIMITED
       FIELDS TERMINATED BY ' '
       LINES TERMINATED BY '\n'
       STORED AS TEXTFILE
       LOCATION '${hiveconf:MH_TMP_DIR}';

INSERT INTO TABLE q25_ctable

SELECT 
    cid 		AS id,
    CASE WHEN 37621 - max(dateid) < 60 
           THEN 1.0 
         ELSE 0.0 END 	AS recency,
    count(oid) 		AS frequency,
    sum(amount) 	AS totalspend
FROM q25_usersegments
GROUP BY cid;


--- CLEANUP--------------------------------------------
DROP TABLE q25_usersegments;



