--Extract competitor product names and model names (if any) from
--online product reviews for a given product.

-- Resources
ADD JAR ${env:BIG_BENCH_QUERIES_DIR}/Resources/opennlp-maxent-3.0.3.jar;
ADD JAR ${env:BIG_BENCH_QUERIES_DIR}/Resources/opennlp-tools-1.5.3.jar;
ADD JAR ${env:BIG_BENCH_QUERIES_DIR}/Resources/bigbenchqueriesmr.jar;
CREATE TEMPORARY FUNCTION find_company AS 'de.bankmark.bigbench.queries.q27.CompanyUDF';

-- !echo Extract competitor product names and model names (if any) from online product reviews for a given product: (item_sk: '${hiveconf:q27_pr_item_sk}');

--Result  --------------------------------------------------------------------
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;

--CREATE RESULT TABLE. Store query result externally in output_dir/qXXresult/
DROP TABLE IF EXISTS ${hiveconf:RESULT_TABLE};
CREATE TABLE ${hiveconf:RESULT_TABLE} (
  pr_review_sk    BIGINT,
  pr_item_sk      BIGINT,
  company_name    STRING,
  review_sentence STRING
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'
STORED AS ${env:BIG_BENCH_hive_default_fileformat_result_table} LOCATION '${hiveconf:RESULT_DIR}';

-- the real query part
INSERT INTO TABLE ${hiveconf:RESULT_TABLE}
SELECT find_company(pr_review_sk, pr_item_sk, pr_review_content) AS (pr_review_sk, pr_item_sk, company_name, review_sentence)
FROM (
  SELECT pr_review_sk, pr_item_sk, pr_review_content
  FROM product_reviews
  WHERE pr_item_sk = ${hiveconf:q27_pr_item_sk}
) subtable
;
