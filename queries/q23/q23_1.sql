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



DROP VIEW IF EXISTS q23_tmp_inv;
CREATE VIEW q23_tmp_inv AS
SELECT 	w_warehouse_name, 
	w_warehouse_sk, 
	i_item_sk, 
	d_moy, 
	stdev, mean, 
 	CASE mean WHEN 0.0 THEN NULL ELSE stdev/mean END cov
FROM (
	SELECT 	w_warehouse_name, 
		w_warehouse_sk, 
		i_item_sk,
		d_moy, 
		stddev_samp(inv_quantity_on_hand) stdev,
		avg(inv_quantity_on_hand) mean
	FROM inventory inv
	JOIN item i ON inv.inv_item_sk = i.i_item_sk
	JOIN warehouse w ON inv.inv_warehouse_sk = w.w_warehouse_sk
	JOIN date_dim d ON inv.inv_date_sk = d.d_date_sk 
	AND d.d_year = 1998
	GROUP BY w_warehouse_name, 
		w_warehouse_sk, 
		i_item_sk, 
		d_moy
	) foo
WHERE CASE mean WHEN 0.0
           THEN 0.0
           ELSE stdev/mean END > 1.0;

