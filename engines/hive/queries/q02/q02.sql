--"INTEL CONFIDENTIAL"
--Copyright 2015  Intel Corporation All Rights Reserved.
--
--The source code contained or described herein and all documents related to the source code ("Material") are owned by Intel Corporation or its suppliers or licensors. Title to the Material remains with Intel Corporation or its suppliers and licensors. The Material contains trade secrets and proprietary and confidential information of Intel or its suppliers and licensors. The Material is protected by worldwide copyright and trade secret laws and treaty provisions. No part of the Material may be used, copied, reproduced, modified, published, uploaded, posted, transmitted, distributed, or disclosed in any way without Intel's prior express written permission.
--
--No license under any patent, copyright, trade secret or other intellectual property right is granted to or conferred upon you by disclosure or delivery of the Materials, either expressly, by implication, inducement, estoppel or otherwise. Any license under such intellectual property rights must be express and approved by Intel in writing.


--Find the top 30 products that are mostly viewed together with a given
--product in online store. Note that the order of products viewed does not matter.

-- Resources
ADD FILE ${env:BIG_BENCH_QUERIES_DIR}/Resources/bigbenchqueriesmr.jar;

--Result -------------------------------------------------------------------------
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;
--CREATE RESULT TABLE. Store query result externally in output_dir/qXXresult/
DROP TABLE IF EXISTS ${hiveconf:RESULT_TABLE};
CREATE TABLE ${hiveconf:RESULT_TABLE} (
  pid1 BIGINT,
  pid2 BIGINT,
  cnt  BIGINT
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'
STORED AS ${env:BIG_BENCH_hive_default_fileformat_result_table} LOCATION '${hiveconf:RESULT_DIR}';

-- the real query part
INSERT INTO TABLE ${hiveconf:RESULT_TABLE}
SELECT  pid1, pid2, COUNT (*) AS cnt
FROM (
  --Make items basket
  FROM (
    -- Select predicate
    SELECT wcs_user_sk AS cid , wcs_item_sk AS pid
    FROM web_clickstreams
    WHERE wcs_item_sk IS NOT NULL AND wcs_user_sk IS NOT NULL
    CLUSTER BY cid

  ) q02_map_output
  REDUCE q02_map_output.cid, q02_map_output.pid
  USING '${env:BIG_BENCH_JAVA} ${env:BIG_BENCH_java_child_process_xmx} -cp bigbenchqueriesmr.jar io.bigdatabenchmark.v1.queries.q02.Red -ITEM_SET_MAX ${hiveconf:q02_NPATH_ITEM_SET_MAX} '
  AS (pid1 BIGINT, pid2 BIGINT)
) q02_temp_basket
WHERE pid1 in ( ${hiveconf:q02_pid1_IN} )
GROUP BY pid1, pid2
CLUSTER BY pid1 ,cnt ,pid2
LIMIT  ${hiveconf:q02_limit}
;
