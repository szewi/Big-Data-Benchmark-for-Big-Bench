

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
