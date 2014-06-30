--Perform category affinity analysis for products viewed together.
-- Resources
ADD FILE ${hiveconf:QUERY_DIR}/mapper_q30.py;
ADD FILE ${hiveconf:QUERY_DIR}/reducer_q30.py;
ADD FILE ${hiveconf:QUERY_DIR}/mapper2_q30.py;
ADD FILE ${hiveconf:QUERY_DIR}/reducer2_q30.py;


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
SELECT 
	ro2.item_sk, 
	ro2.affine_item_sk, 
	ro2.item, 
	ro2.affine_item, 
	ro2.frequency 
FROM (
	FROM (
		SELECT concat(	ro.item_id, 
					':',
					ro.affine_item_id) 	AS combined_key,
			ro.item_id 				AS item_id, 
			ro.item 					AS item, 
			ro.affine_item_id 			AS affine_item_id, 
			ro.affine_item 			AS affine_item,
			1 						AS frequency
		FROM (
			FROM (
				SELECT concat(	wcs.wcs_user_sk, 
						':', 
						wcs.wcs_click_date_sk) 	AS key,
					i.i_item_sk 				AS item_sk,
					i.i_item_id 				AS item
				FROM web_clickstreams wcs 
				JOIN item i ON wcs.wcs_item_sk = i.i_item_sk
				AND wcs.wcs_user_sk IS NOT NULL
				AND wcs.wcs_item_sk IS NOT NULL
				--MAP 
				--	wcs.wcs_user_sk, 
				--	wcs.wcs_click_date_sk, 
				--	i.i_item_sk, 
				--	i.i_item_id
				--USING 'python mapper_q30.py'  --(val1 + ":" + val2, val2, val3) --concat(string|binary A, string|binary B...)
				--AS 	key, 
				--	item_sk, 
				--	item
				CLUSTER BY key
			) mo
			REDUCE mo.key, mo.item_sk, mo.item
			USING 'python reducer_q30.py'
			AS (item_id, item, affine_item_id,	affine_item)
		) ro
		--MAP 	ro.item_id, 
		--	ro.item, 
		--	ro.affine_item_id, 
		--	ro.affine_item
		--USING 'python mapper2_q30.py' --(val1 + ":" + val3, val1, val2, val3, val4, 1)	
		--AS combined_key, item_id, item, affine_item_id, affine_item, frequency
		--CLUSTER BY combined_key
	) mo2
	REDUCE mo2.combined_key, mo2.item_id, mo2.item, mo2.affine_item_id, mo2.affine_item, mo2.frequency
	USING 'python reducer2_q30.py'
	AS (	item_sk, 
		item, 
		affine_item_sk, 
		affine_item, 
		frequency)
) ro2
CLUSTER BY ro2.frequency
;
