--"INTEL CONFIDENTIAL"
--Copyright 2015  Intel Corporation All Rights Reserved.
--
--The source code contained or described herein and all documents related to the source code ("Material") are owned by Intel Corporation or its suppliers or licensors. Title to the Material remains with Intel Corporation or its suppliers and licensors. The Material contains trade secrets and proprietary and confidential information of Intel or its suppliers and licensors. The Material is protected by worldwide copyright and trade secret laws and treaty provisions. No part of the Material may be used, copied, reproduced, modified, published, uploaded, posted, transmitted, distributed, or disclosed in any way without Intel's prior express written permission.
--
--No license under any patent, copyright, trade secret or other intellectual property right is granted to or conferred upon you by disclosure or delivery of the Materials, either expressly, by implication, inducement, estoppel or otherwise. Any license under such intellectual property rights must be express and approved by Intel in writing.


--Find the last 5 products that are mostly viewed before a given product
--was purchased online. Only products in certain categories and viewed within 10
--days before the purchase date are considered.

-- Resources
--ADD FILE ${hiveconf:QUERY_DIR}/mapper_q3.py;
ADD FILE ${hiveconf:QUERY_DIR}/reducer_q3.py;

--Result -------------------------------------------------------------------------
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;
--CREATE RESULT TABLE. Store query result externally in output_dir/qXXresult/
DROP TABLE IF EXISTS ${hiveconf:RESULT_TABLE};
CREATE TABLE ${hiveconf:RESULT_TABLE} (
  lastviewed_item BIGINT,
  purchased_item  BIGINT,
  cnt             BIGINT
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'
STORED AS ${env:BIG_BENCH_hive_default_fileformat_result_table} LOCATION '${hiveconf:RESULT_DIR}';

-- the real query part
INSERT INTO TABLE ${hiveconf:RESULT_TABLE}
SELECT lastviewed_item, purchased_item, count(*)
FROM (
  FROM (
    SELECT
      wcs_user_sk       AS user,
      wcs_click_date_sk AS lastviewed_date,
      wcs_item_sk       AS lastviewed_item,
      wcs_sales_sk      AS lastviewed_sale
    FROM web_clickstreams w
    -- only select clickstreams resulting in a purchase user_sk = null -> only non buying visitor
    WHERE wcs_user_sk IS NOT NULL
    CLUSTER BY user
  ) q03_map_output
  REDUCE
  q03_map_output.user,
  q03_map_output.lastviewed_date,
  q03_map_output.lastviewed_item,
  q03_map_output.lastviewed_sale
  --Reducer script selects only products viewed within 'q03_days_before_purchase' days before the purchase date
  USING 'python reducer_q3.py ${hiveconf:q03_days_before_purchase}'
  AS (lastviewed_item BIGINT, purchased_item BIGINT)
) q03_nPath
join item i on (i.i_item_sk = q03_nPath.lastviewed_item
  --Only products in certain categories
  AND i.i_category_id IN (${hiveconf:q03_purchased_item_category_IN})
)
WHERE purchased_item IN ( ${hiveconf:q03_purchased_item_IN} )
GROUP BY lastviewed_item, purchased_item,i_category
;
