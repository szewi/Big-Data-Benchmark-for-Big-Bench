--Perform category afinity analysis for products purchased online together.


-- Resources
ADD FILE ${hiveconf:QUERY_DIR}/mapper_q29.py;
ADD FILE ${hiveconf:QUERY_DIR}/reducer_q29.py;
ADD FILE ${hiveconf:QUERY_DIR}/mapper2_q29.py;
ADD FILE ${hiveconf:QUERY_DIR}/reducer2_q29.py;

-- Resources

--Result  --------------------------------------------------------------------		
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;
--CREATE RESULT TABLE. Store query result externally in output_dir/qXXresult/
DROP TABLE IF EXISTS ${hiveconf:RESULT_TABLE};
CREATE TABLE ${hiveconf:RESULT_TABLE}
ROW FORMAT
DELIMITED FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
STORED AS ${env:BIG_BENCH_hive_default_fileformat_result_table}
LOCATION '${hiveconf:RESULT_DIR}' 
AS
-- Begin: the real query part
SELECT 	
	ro2.category_id, 
	ro2.affine_category_id, 
	ro2.category, 
	ro2.affine_category, 
	ro2.frequency 
FROM 
(
	FROM 
	(
		FROM 
		(
			FROM 
			(
				SELECT	ws.ws_order_number 	AS ordernumber, 
						i.i_category_id 	AS category_id, 
						i.i_category		AS category
				FROM web_sales ws 
				JOIN item i ON (ws.ws_item_sk = i.i_item_sk AND i.i_category_id IS NOT NULL)
				CLUSTER BY ordernumber
			) mo
			REDUCE 	
				mo.ordernumber, 
				mo.category_id, 
				mo.category
			USING 'python reducer_q29.py'
			AS 	(category_id,
				 category, 
				 affine_category_id, 
				 affine_category )
		) ro
		MAP 	ro.category_id, 
			ro.category, 
			ro.affine_category_id, 
			ro.affine_category
		USING 'python mapper2_q29.py'
		AS 	combined_key, 
			category_id, 
			category, 
			affine_category_id, 
			affine_category, 
			frequency
		CLUSTER BY combined_key
	) mo2
	REDUCE 	
		mo2.combined_key, 
		mo2.category_id, 
		mo2.category, 
		mo2.affine_category_id, 
		mo2.affine_category, 
		mo2.frequency
	USING 'python reducer2_q29.py'
	AS (	category_id, 
		category, 
		affine_category_id, 
	  	affine_category, 
		frequency)
) ro2
CLUSTER BY ro2.frequency
;
