--"INTEL CONFIDENTIAL"
--Copyright 2015  Intel Corporation All Rights Reserved.
--
--The source code contained or described herein and all documents related to the source code ("Material") are owned by Intel Corporation or its suppliers or licensors. Title to the Material remains with Intel Corporation or its suppliers and licensors. The Material contains trade secrets and proprietary and confidential information of Intel or its suppliers and licensors. The Material is protected by worldwide copyright and trade secret laws and treaty provisions. No part of the Material may be used, copied, reproduced, modified, published, uploaded, posted, transmitted, distributed, or disclosed in any way without Intel's prior express written permission.
--
--No license under any patent, copyright, trade secret or other intellectual property right is granted to or conferred upon you by disclosure or delivery of the Materials, either expressly, by implication, inducement, estoppel or otherwise. Any license under such intellectual property rights must be express and approved by Intel in writing.




-- Resources
-------------------------------------------------------------------------------
--Part 1:
-------------------------------------------------------------------------------
DROP TABLE IF EXISTS q30_c_affinity_input;


CREATE TABLE q30_c_affinity_input AS
SELECT 
  i.i_category_id   AS category_cd,
  s.wcs_user_sk     AS customer_id,
  i.i_item_sk     AS item
FROM web_clickstreams s 
JOIN item i ON s.wcs_item_sk = i.i_item_sk
WHERE s.wcs_item_sk   IS NOT NULL
  AND i.i_category_id   IS NOT NULL
  AND s.wcs_user_sk   IS NOT NULL;
