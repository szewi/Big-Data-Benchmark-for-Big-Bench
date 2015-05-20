--"INTEL CONFIDENTIAL"
--Copyright 2015  Intel Corporation All Rights Reserved.
--
--The source code contained or described herein and all documents related to the source code ("Material") are owned by Intel Corporation or its suppliers or licensors. Title to the Material remains with Intel Corporation or its suppliers and licensors. The Material contains trade secrets and proprietary and confidential information of Intel or its suppliers and licensors. The Material is protected by worldwide copyright and trade secret laws and treaty provisions. No part of the Material may be used, copied, reproduced, modified, published, uploaded, posted, transmitted, distributed, or disclosed in any way without Intel's prior express written permission.
--
--No license under any patent, copyright, trade secret or other intellectual property right is granted to or conferred upon you by disclosure or delivery of the Materials, either expressly, by implication, inducement, estoppel or otherwise. Any license under such intellectual property rights must be express and approved by Intel in writing.


--Perform category affinity analysis for products viewed together.

-- Resources
ADD FILE ${hiveconf:QUERY_DIR}/reducer_q30.py;
--ADD FILE ${env:BIG_BENCH_QUERIES_DIR}/Resources/bigbenchqueriesmr.jar; 

--Result  --------------------------------------------------------------------
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;

--CREATE RESULT TABLE. Store query result externally in output_dir/qXXresult/
DROP TABLE IF EXISTS ${hiveconf:RESULT_TABLE};
CREATE TABLE ${hiveconf:RESULT_TABLE} (
  category_id        STRING,
  affine_category_id STRING,
  category           STRING,
  affine_category    STRING,
  frequency          BIGINT
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'
STORED AS ${env:BIG_BENCH_hive_default_fileformat_result_table} LOCATION '${hiveconf:RESULT_DIR}';

-- Begin: the real query part
INSERT INTO TABLE ${hiveconf:RESULT_TABLE}
SELECT
  ro.category_id AS category_id,
  ro.affine_category_id AS affine_category_id,
  ro.category AS category,
  ro.affine_category AS affine_category,
  count(*) as frequency
FROM (
  FROM (
    SELECT
      concat(wcs.wcs_user_sk, ':', wcs.wcs_click_date_sk) AS combined_key,
      i.i_category_id AS category_id,
      i.i_category AS category
    FROM web_clickstreams wcs
    JOIN item i ON (wcs.wcs_item_sk = i.i_item_sk AND i.i_category_id IS NOT NULL)
    AND wcs.wcs_user_sk IS NOT NULL
    AND wcs.wcs_item_sk IS NOT NULL
    CLUSTER BY combined_key
  ) mo
  REDUCE
    mo.combined_key,
    mo.category_id,
    mo.category
  USING 'python reducer_q30.py'
  --USING '${env:BIG_BENCH_JAVA} ${env:BIG_BENCH_java_child_process_xmx} -cp bigbenchqueriesmr.jar io.bigdatabenchmark.v1.queries.q30.Red'
  AS (
    category_id,
    category,
    affine_category_id,
    affine_category )
) ro
GROUP BY ro.category_id , ro.affine_category_id, ro.category ,ro.affine_category
CLUSTER BY frequency
;
