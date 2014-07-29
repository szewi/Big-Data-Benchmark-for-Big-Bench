--Find products that are sold together frequently in given
--stores. Only products in certain categories sold in specific stores are considered,
--and "sold together frequently" means at least 50 customers bought these products 
--together in a transaction.

-- Resources
--add file ${env:BIG_BENCH_HIVE_LIBS}/hive-contrib.jar;
add file ${env:BIG_BENCH_QUERIES_DIR}/Resources/bigbenchqueriesmr.jar;



--Result -------------------------------------------------------------------------
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;
--CREATE RESULT TABLE. Store query result externally in output_dir/qXXresult/
DROP TABLE IF EXISTS ${hiveconf:RESULT_TABLE};
CREATE TABLE ${hiveconf:RESULT_TABLE}
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'
STORED AS ${env:BIG_BENCH_hive_default_fileformat_result_table} LOCATION '${hiveconf:RESULT_DIR}' 
AS
-- the real query part
--Find the most frequent ones
SELECT	pid1, pid2, COUNT (*) AS cnt
FROM (
	--Make items basket
	FROM (
		-- Joining two tables
		SELECT s.ss_ticket_number AS oid , s.ss_item_sk AS pid
		FROM store_sales s
		INNER JOIN item i ON (s.ss_item_sk = i.i_item_sk)
		WHERE i.i_category_id in (${hiveconf:q01_i_category_id_IN}) 
		AND s.ss_store_sk in (${hiveconf:q01_ss_store_sk_IN})
		CLUSTER BY oid

	) q01_map_output
	REDUCE q01_map_output.oid, q01_map_output.pid
	USING 'java ${env:BIG_BENCH_java_child_process_xmx} -cp bigbenchqueriesmr.jar de.bankmark.bigbench.queries.q01.Red -ITEM_SET_MAX ${hiveconf:q01_NPATH_ITEM_SET_MAX} '
	AS (pid1 BIGINT, pid2 BIGINT)
) q01_temp_basket
GROUP BY pid1, pid2
HAVING COUNT (pid1) > ${hiveconf:q01_COUNT_pid1_greater}
CLUSTER BY pid1 ,cnt ,pid2
;
