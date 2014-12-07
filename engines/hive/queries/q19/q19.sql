--Retrieve the items with the highest number of returns where the num-
--ber of returns was approximately equivalent across all store and web channels
--(within a tolerance of +/- 10%), within the week ending a given date. Analyze
--the online reviews for these items to see if there are any major negative reviews.

-- Resources

-----Store returns in date range q19_tmp_date1-------------------------------------------------
DROP VIEW IF EXISTS ${hiveconf:TEMP_TABLE1};
CREATE VIEW ${hiveconf:TEMP_TABLE1} AS
SELECT
  i_item_sk item_id,
  SUM(sr_return_quantity) sr_item_qty
FROM
  store_returns sr
INNER JOIN item i ON sr.sr_item_sk = i.i_item_sk
INNER JOIN date_dim d ON sr.sr_returned_date_sk = d.d_date_sk
INNER JOIN (
  SELECT d1.d_date_sk
  FROM date_dim d1
  LEFT SEMI JOIN date_dim d2 ON (
    d1.d_week_seq = d2.d_week_seq
    AND d2.d_date IN ( ${hiveconf:q19_storeReturns_date_IN} )
  )
) d1 ON sr.sr_returned_date_sk = d1.d_date_sk
GROUP BY i_item_sk
HAVING SUM(sr_return_quantity) > 0
;


-----Web returns in daterange q19_tmp_date2 ------------------------------------------------------
DROP VIEW IF EXISTS ${hiveconf:TEMP_TABLE2};
--make ${hiveconf:TEMP_TABLE2}
CREATE VIEW ${hiveconf:TEMP_TABLE2} AS
SELECT
  i_item_sk item_id,
  SUM(wr_return_quantity) wr_item_qty
FROM
  web_returns wr
INNER JOIN item i ON wr.wr_item_sk = i.i_item_sk
INNER JOIN date_dim d ON wr.wr_returned_date_sk = d.d_date_sk
INNER JOIN (
  SELECT d1.d_date_sk
  FROM date_dim d1
  LEFT SEMI JOIN date_dim d2 ON (
    d1.d_week_seq = d2.d_week_seq 
    AND d2.d_date IN ( ${hiveconf:q19_webReturns_date_IN} )
  )
) d2 ON wr.wr_returned_date_sk = d2.d_date_sk
GROUP BY i_item_sk
HAVING SUM(wr_return_quantity) > 0
;


----return items -------------------------------------------------------------------
DROP VIEW IF EXISTS ${hiveconf:TEMP_TABLE3};
--make ${hiveconf:TEMP_TABLE3}
CREATE VIEW ${hiveconf:TEMP_TABLE3} AS
SELECT
  st.item_id item,
  sr_item_qty,
  100.0 * sr_item_qty / (sr_item_qty+wr_item_qty) / 2.0 sr_dev,
  wr_item_qty,
  100.0 * wr_item_qty / (sr_item_qty+wr_item_qty) / 2.0 wr_dev,
  (sr_item_qty + wr_item_qty) / 2.0 average
FROM ${hiveconf:TEMP_TABLE1} st
INNER JOIN ${hiveconf:TEMP_TABLE2} wt ON st.item_id = wt.item_id
--CLUSTER BY average desc
DISTRIBUTE BY average
SORT BY average DESC
LIMIT ${hiveconf:q19_store_return_limit}
;

---Sentiment analysis and Result----------------------------------------------------------------
--- we can reuse the  sentiment analysis helper class from q10
ADD JAR ${env:BIG_BENCH_QUERIES_DIR}/Resources/opennlp-maxent-3.0.3.jar;
ADD JAR ${env:BIG_BENCH_QUERIES_DIR}/Resources/opennlp-tools-1.5.3.jar;
ADD JAR ${env:BIG_BENCH_QUERIES_DIR}/Resources/bigbenchqueriesmr.jar;
CREATE TEMPORARY FUNCTION extract_sentiment AS 'de.bankmark.bigbench.queries.q10.SentimentUDF';


--Result  returned items with negative sentiment --------------------------------------------
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;

--CREATE RESULT TABLE. Store query result externally in output_dir/qXXresult/
DROP TABLE IF EXISTS ${hiveconf:RESULT_TABLE};
CREATE TABLE ${hiveconf:RESULT_TABLE} (
  pr_item_sk      BIGINT,
  review_sentence STRING,
  sentiment       STRING,
  sentiment_word  STRING
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'
STORED AS ${env:BIG_BENCH_hive_default_fileformat_result_table} LOCATION '${hiveconf:RESULT_DIR}';

---- the real query --------------
INSERT INTO TABLE ${hiveconf:RESULT_TABLE}
SELECT *
FROM
(
  SELECT extract_sentiment(pr.pr_item_sk, pr.pr_review_content) AS (
    pr_item_sk,
    review_sentence,
    sentiment,
    sentiment_word
  )
  FROM product_reviews pr
  LEFT SEMI JOIN ${hiveconf:TEMP_TABLE3} ri ON pr.pr_item_sk = ri.item
) q19_tmp_sentiment
WHERE q19_tmp_sentiment.sentiment = 'NEG ';


--- cleanup---------------------------------------------------------------------------


DROP VIEW IF EXISTS ${hiveconf:TEMP_TABLE1};
DROP VIEW IF EXISTS ${hiveconf:TEMP_TABLE2};
DROP VIEW IF EXISTS ${hiveconf:TEMP_TABLE3};
