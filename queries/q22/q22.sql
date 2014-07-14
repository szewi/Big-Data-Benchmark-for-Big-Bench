--For all items whose price was changed on a given date,
--compute the percentage change in inventory between the 30-day period BEFORE 
--the price change and the 30-day period AFTER the change. Group this
--information by warehouse.

-- Resources


--DROP TABLE IF EXISTS q22_coalition_22;
--CREATE TABLE q22_coalition_22 AS
--  SELECT *
--    FROM inventory inv
--    JOIN (SELECT * 
--            FROM item i 
--           WHERE i.i_current_price > 0.98 
--             AND i.i_current_price < 1.5) i
--         ON inv.inv_item_sk = i.i_item_sk
--    JOIN warehouse w ON inv.inv_warehouse_sk = w.w_warehouse_sk
--    JOIN date_dim d ON inv.inv_date_sk = d.d_date_sk
--   WHERE datediff(d_date, '2000-05-08') >= -30 
--    AND datediff(d_date, '2000-05-08') <= 30;


DROP TABLE IF EXISTS ${hiveconf:TEMP_TABLE1};
CREATE TABLE ${hiveconf:TEMP_TABLE1} AS
SELECT 	w_warehouse_name, 
	i_item_id,
	 sum(CASE WHEN datediff(d_date, '${hiveconf:q22_date}') < 0 
	          THEN inv_quantity_on_hand 
	          ELSE 0 END) AS inv_before,
	 sum(CASE WHEN datediff(d_date, '${hiveconf:q22_date}') >= 0 
	          THEN inv_quantity_on_hand
	          ELSE 0 END) AS inv_after
FROM (
	SELECT *
	    FROM inventory inv
	    JOIN (SELECT * 
		    FROM item i 
		   WHERE i.i_current_price > ${hiveconf:q22_i_current_price_min}
		     AND i.i_current_price < ${hiveconf:q22_i_current_price_max}
		) i
		 ON inv.inv_item_sk = i.i_item_sk
	    JOIN warehouse w ON inv.inv_warehouse_sk = w.w_warehouse_sk
	    JOIN date_dim d ON inv.inv_date_sk = d.d_date_sk
	   WHERE datediff(d_date, '${hiveconf:q22_date}') >= -30 
	     AND datediff(d_date, '${hiveconf:q22_date}') <= 30

)q22_coalition_22 
GROUP BY w_warehouse_name, i_item_id;


DROP TABLE IF EXISTS ${hiveconf:TEMP_TABLE2};
CREATE TABLE ${hiveconf:TEMP_TABLE2} AS
  SELECT w_warehouse_name, inv_after/inv_before
    FROM ${hiveconf:TEMP_TABLE1}
   WHERE inv_before > 0
     AND inv_after/inv_before >= 2.0/3.0 
     AND inv_after/inv_before <= 3.0/2.0;


--Result --------------------------------------------------------------------		
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
  SELECT name.w_warehouse_name, i_item_id, inv_before, inv_after
    FROM ${hiveconf:TEMP_TABLE1} name 
    JOIN ${hiveconf:TEMP_TABLE2} nombre 
      ON name.w_warehouse_name = nombre.w_warehouse_name
   ORDER BY w_warehouse_name, i_item_id;


---- cleanup ----------------
DROP TABLE IF EXISTS ${hiveconf:TEMP_TABLE2};
DROP TABLE IF EXISTS ${hiveconf:TEMP_TABLE1};
--DROP TABLE IF EXISTS q22_coalition_22;
