--"INTEL CONFIDENTIAL"
--Copyright 2015  Intel Corporation All Rights Reserved.
--
--The source code contained or described herein and all documents related to the source code ("Material") are owned by Intel Corporation or its suppliers or licensors. Title to the Material remains with Intel Corporation or its suppliers and licensors. The Material contains trade secrets and proprietary and confidential information of Intel or its suppliers and licensors. The Material is protected by worldwide copyright and trade secret laws and treaty provisions. No part of the Material may be used, copied, reproduced, modified, published, uploaded, posted, transmitted, distributed, or disclosed in any way without Intel's prior express written permission.
--
--No license under any patent, copyright, trade secret or other intellectual property right is granted to or conferred upon you by disclosure or delivery of the Materials, either expressly, by implication, inducement, estoppel or otherwise. Any license under such intellectual property rights must be express and approved by Intel in writing.


--For online sales, compare the total sales in which customers checked
--online reviews before making the purchase and that of sales in which customers
--did not read reviews. Consider only online sales for a specific category in a given
--year.

-- Resources
ADD FILE ${hiveconf:QUERY_DIR}/q8_reducer.py;

CREATE VIEW IF NOT EXISTS ${hiveconf:TEMP_TABLE1} AS
SELECT d_date_sk
FROM date_dim d
WHERE d.d_date >= '${hiveconf:q08_startDate}'
AND   d.d_date <= '${hiveconf:q08_endDate}'
;
---- !echo "created ${hiveconf:TEMP_TABLE1}";

--PART 1 - sales that users have viewed the review pages--------------------------------------------------------
DROP VIEW IF EXISTS ${hiveconf:TEMP_TABLE2};
CREATE VIEW IF not exists ${hiveconf:TEMP_TABLE2} AS
SELECT DISTINCT s_sk
FROM (
  FROM (
    SELECT
      c.wcs_user_sk       AS uid,
      c.wcs_click_date_sk AS c_date,
      c.wcs_click_time_sk AS c_time,
      c.wcs_sales_sk      AS sales_sk,
      w.wp_type           AS wpt
    FROM web_clickstreams c
    JOIN ${hiveconf:TEMP_TABLE1} d ON (c.wcs_click_date_sk = d.d_date_sk)
    INNER JOIN web_page w ON c.wcs_web_page_sk = w.wp_web_page_sk
    WHERE c.wcs_user_sk IS NOT NULL
    CLUSTER BY uid
  ) q08_map_output
  REDUCE q08_map_output.uid,
    q08_map_output.c_date,
    q08_map_output.c_time,
    q08_map_output.sales_sk,
    q08_map_output.wpt
  USING 'python q8_reducer.py ${hiveconf:q08_category}'
  AS (s_date BIGINT, s_sk BIGINT)
) q08npath
;


--PART 2 - helper table: sales within one year starting 1999-09-02  ---------------------------------------
DROP VIEW IF EXISTS ${hiveconf:TEMP_TABLE3};
CREATE VIEW IF NOT EXISTS ${hiveconf:TEMP_TABLE3} AS
SELECT ws_net_paid, ws_order_number
FROM web_sales ws
JOIN ${hiveconf:TEMP_TABLE1} d ON ( ws.ws_sold_date_sk = d.d_date_sk)
;


--PART 3 - for sales in given year, compute sales in which customers checked online reviews vs. sales in which customers did not read reviews.
--Result  --------------------------------------------------------------------
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;
--CREATE RESULT TABLE. Store query result externally in output_dir/qXXresult/
DROP TABLE IF EXISTS ${hiveconf:RESULT_TABLE};
CREATE TABLE ${hiveconf:RESULT_TABLE} (
  q08_review_sales_amount    BIGINT,
  no_q08_review_sales_amount BIGINT
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'
STORED AS ${env:BIG_BENCH_hive_default_fileformat_result_table} LOCATION '${hiveconf:RESULT_DIR}';

-- the real query part----------------------------------------------------------------------
INSERT INTO TABLE ${hiveconf:RESULT_TABLE}
SELECT
  q08_review_sales.amount AS q08_review_sales_amount,
  q08_all_sales.amount - q08_review_sales.amount AS no_q08_review_sales_amount
FROM (
  SELECT 1 AS id, SUM(ws_net_paid) as amount
  FROM ${hiveconf:TEMP_TABLE3} ws
  INNER JOIN ${hiveconf:TEMP_TABLE2} sr ON ws.ws_order_number = sr.s_sk
) q08_review_sales
JOIN (
  SELECT 1 AS id, SUM(ws_net_paid) as amount
  FROM ${hiveconf:TEMP_TABLE3} ws
)  q08_all_sales
ON q08_review_sales.id = q08_all_sales.id
;


--cleanup-------------------------------------------------------------------
DROP VIEW IF EXISTS ${hiveconf:TEMP_TABLE2};
DROP VIEW IF EXISTS ${hiveconf:TEMP_TABLE3};
DROP VIEW IF EXISTS ${hiveconf:TEMP_TABLE1};
