--Find all customers who viewed items of a given category on the web
--in a given month and year that was followed by an in-store purchase of an item from the same category in the three
--consecutive months.

-- Resources

--Result  --------------------------------------------------------------------
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;

DROP TABLE IF EXISTS ${hiveconf:RESULT_TABLE};
CREATE TABLE ${hiveconf:RESULT_TABLE} (
  c_date BIGINT,
  s_date BIGINT,
  i_id    BIGINT,
  u_id    BIGINT
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'
STORED AS ${env:BIG_BENCH_hive_default_fileformat_result_table}  LOCATION '${hiveconf:RESULT_DIR}';

INSERT INTO TABLE ${hiveconf:RESULT_TABLE}
-- the real query part
SELECT DISTINCT wcsView.wcs_click_date_sk, 
                storeView.ss_sold_date_sk,
                wcsView.wcs_item_sk,
                wcsView.wcs_user_sk
FROM( 
  SELECT wcs.wcs_item_sk,
         wcs.wcs_user_sk,
	 wcs.wcs_click_date_sk,
	 i.i_category
  FROM web_clickstreams wcs
-- filter given month and year 
  LEFT SEMI JOIN (
    SELECT d_date_sk
    FROM date_dim d
    WHERE d.d_date >= '${hiveconf:q12_startDate}'
    AND   d.d_date <= '${hiveconf:q12_endDate1}'
  ) dd ON ( wcs.wcs_click_date_sk=dd.d_date_sk )
-- filter given category 
  JOIN item i ON (wcs.wcs_item_sk = i.i_item_sk AND i.i_category IN (${hiveconf:q12_i_category_IN}) )
  WHERE wcs.wcs_user_sk IS NOT NULL
) wcsView
JOIN( 
  SELECT ss.ss_item_sk,
         ss.ss_customer_sk,
         ss.ss_sold_date_sk,
	 i.i_category
  FROM store_sales ss
  LEFT SEMI JOIN (
    SELECT d_date_sk
    FROM date_dim d
-- filter given month and year + 3 consecutive months
    WHERE d.d_date >= '${hiveconf:q12_startDate}'
    AND   d.d_date <= '${hiveconf:q12_endDate2}'
  ) dd ON ( ss.ss_sold_date_sk=dd.d_date_sk )
-- filter given category 
  JOIN item i ON (ss.ss_item_sk = i.i_item_sk AND i.i_category IN (${hiveconf:q12_i_category_IN})) 
  WHERE ss.ss_customer_sk IS NOT NULL
) storeView
ON (wcs_user_sk = ss_customer_sk)
-- filter 3 consecutive months: buy AFTER view on website
WHERE wcs_click_date_sk < ss_sold_date_sk
AND wcsView.i_category = storeView.i_category
CLUSTER BY wcs_click_date_sk,
           wcs_item_sk,
	   wcs_user_sk 
;
