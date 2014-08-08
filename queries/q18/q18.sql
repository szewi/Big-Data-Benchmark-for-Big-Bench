--Identify the stores with flat or declining sales in 3 consecutive months,
--check if there are any negative reviews regarding these stores available online.

-- Resources
ADD JAR ${env:BIG_BENCH_QUERIES_DIR}/Resources/opennlp-maxent-3.0.3.jar;
ADD JAR ${env:BIG_BENCH_QUERIES_DIR}/Resources/opennlp-tools-1.5.3.jar;
ADD JAR ${env:BIG_BENCH_QUERIES_DIR}/Resources/bigbenchqueriesmr.jar;
CREATE TEMPORARY FUNCTION extract_NegSentiment AS 'de.bankmark.bigbench.queries.q18.NegativeSentimentUDF';

DROP TABLE IF EXISTS ${hiveconf:TEMP_TABLE};
CREATE TABLE ${hiveconf:TEMP_TABLE} (
  s_store_name       STRING,
  pr_review_date     STRING,
  pr_review_content  STRING
);

INSERT INTO TABLE ${hiveconf:TEMP_TABLE}
SELECT
  s_store_name,
  pr_review_date,
  pr_review_content
FROM (
  --select store_name for stores with flat or declining sales in 3 consecutive months.
  SELECT s_store_name
  FROM store s
  JOIN (
    -- linear regression part
    SELECT
      temp.cat AS cat,
      --SUM(temp.x)as sumX,
      --SUM(temp.y)as sumY,
      --SUM(temp.xy)as sumXY,
      --SUM(temp.xx)as sumXSquared,
      --count(temp.x) as N,
      --N * sumXY - sumX * sumY AS numerator,
      --N * sumXSquared - sumX*sumX AS denom
      --numerator / denom as slope,
      --(sumY - slope * sumX) / N as intercept
      --(count(temp.x) * SUM(temp.xy) - SUM(temp.x) * SUM(temp.y)) AS numerator,
      --(count(temp.x) * SUM(temp.xx) - SUM(temp.x) * SUM(temp.x)) AS denom
      --numerator / denom as slope,
      --(sumY - slope * sumX) / N as intercept
      ((count(temp.x) * SUM(temp.xy) - SUM(temp.x) * SUM(temp.y)) / (count(temp.x) * SUM(temp.xx) - SUM(temp.x) * SUM(temp.x)) ) as slope,
      (SUM(temp.y) - ((count(temp.x) * SUM(temp.xy) - SUM(temp.x) * SUM(temp.y)) / (count(temp.x) * SUM(temp.xx) - SUM(temp.x) * SUM(temp.x)) ) * SUM(temp.x)) / count(temp.x) as intercept
    FROM (
      SELECT
        s.ss_store_sk AS cat,
        s.ss_sold_date_sk  AS x,
        SUM(s.ss_net_paid) AS y,
        s.ss_sold_date_sk * SUM(s.ss_net_paid) AS xy,
        s.ss_sold_date_sk*s.ss_sold_date_sk AS xx
      FROM store_sales s
      --select date range
      LEFT SEMI JOIN (
        SELECT d_date_sk
        FROM date_dim d
        WHERE d.d_date >= '${hiveconf:q18_startDate}'
        AND   d.d_date <= '${hiveconf:q18_endDate}'
      ) dd ON ( s.ss_sold_date_sk=dd.d_date_sk )
      WHERE s.ss_store_sk <= 18
      GROUP BY s.ss_store_sk, s.ss_sold_date_sk
    ) temp
    GROUP BY temp.cat
  ) c on s.s_store_sk = c.cat
  WHERE c.slope < 0
) tmp
JOIN  product_reviews pr on (true)
WHERE instr(pr.pr_review_content, tmp.s_store_name) > 0
;


--Result  --------------------------------------------------------------------
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;
--Prepare result storage
DROP TABLE IF EXISTS ${hiveconf:RESULT_TABLE};
CREATE TABLE ${hiveconf:RESULT_TABLE} (
  s_store_name    STRING,
  review_date     STRING,
  review_sentence STRING,
  sentiment       STRING,
  sentiment_word  STRING
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'
STORED AS ${env:BIG_BENCH_hive_default_fileformat_result_table} LOCATION '${hiveconf:RESULT_DIR}';

-- the real query
INSERT INTO TABLE ${hiveconf:RESULT_TABLE}
SELECT extract_NegSentiment( s_store_name, pr_review_date, pr_review_content) AS ( s_store_name, review_date, review_sentence, sentiment, sentiment_word )
--select product_reviews containing the name of a store. Consider only stores with flat or declining sales in 3 consecutive months.
FROM ${hiveconf:TEMP_TABLE}
;


-- Cleanup ----------------------------------------
DROP TABLE IF EXISTS ${hiveconf:TEMP_TABLE};
