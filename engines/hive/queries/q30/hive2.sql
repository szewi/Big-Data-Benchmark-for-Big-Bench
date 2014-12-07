

-- Resources
-------------------------------------------------------------------------------
--Part 3
-------------------------------------------------------------------------------

DROP TABLE IF EXISTS q30_c_affinity_out;

CREATE TABLE q30_c_affinity_out (
    customer_id    STRING,
    score          STRING
) STORED AS TEXTFILE;

LOAD DATA INPATH '/mahout_result/thirty/part-r-00000' OVERWRITE INTO TABLE q30_c_affinity_out;
