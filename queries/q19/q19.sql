-- Global hive options (see: Big-Bench/setEnvVars)
--set hive.exec.parallel=${env:BIG_BENCH_hive_exec_parallel};
--set hive.exec.parallel.thread.number=${env:BIG_BENCH_hive_exec_parallel_thread_number};
--set hive.exec.compress.intermediate=${env:BIG_BENCH_hive_exec_compress_intermediate};
--set mapred.map.output.compression.codec=${env:BIG_BENCH_mapred_map_output_compression_codec};
--set hive.exec.compress.output=${env:BIG_BENCH_hive_exec_compress_output};
--set mapred.output.compression.codec=${env:BIG_BENCH_mapred_output_compression_codec};
--set hive.default.fileformat=${env:BIG_BENCH_hive_default_fileformat};
--set hive.optimize.mapjoin.mapreduce=${env:BIG_BENCH_hive_optimize_mapjoin_mapreduce};
--set hive.optimize.bucketmapjoin=${env:BIG_BENCH_hive_optimize_bucketmapjoin};
--set hive.optimize.bucketmapjoin.sortedmerge=${env:BIG_BENCH_hive_optimize_bucketmapjoin_sortedmerge};
--set hive.auto.convert.join=${env:BIG_BENCH_hive_auto_convert_join};
--set hive.auto.convert.sortmerge.join=${env:BIG_BENCH_hive_auto_convert_sortmerge_join};
--set hive.auto.convert.sortmerge.join.noconditionaltask=${env:BIG_BENCH_hive_auto_convert_sortmerge_join_noconditionaltask};
--set hive.optimize.ppd=${env:BIG_BENCH_hive_optimize_ppd};
--set hive.optimize.index.filter=${env:BIG_BENCH_hive_optimize_index_filter};

--display settings
set hive.exec.parallel;
set hive.exec.parallel.thread.number;
set hive.exec.compress.intermediate;
set mapred.map.output.compression.codec;
set hive.exec.compress.output;
set mapred.output.compression.codec;
set hive.default.fileformat;
set hive.optimize.mapjoin.mapreduce;
set hive.mapjoin.smalltable.filesize;
set hive.optimize.bucketmapjoin;
set hive.optimize.bucketmapjoin.sortedmerge;
set hive.auto.convert.join;
set hive.auto.convert.sortmerge.join;
set hive.auto.convert.sortmerge.join.noconditionaltask;
set hive.optimize.ppd;
set hive.optimize.index.filter;
-- Database
use ${env:BIG_BENCH_HIVE_DATABASE};

-- Resources
--ADD FILE ${hiveconf:QUERY_DIR}/mapper_19.py;
--ADD FILE ${hiveconf:QUERY_DIR}/reducer_19.py;


-- Result file configuration
			  
-------------------------------------------------------------------------------


-----Store returns in date range q19_tmp_date1-------------------------------------------------
DROP VIEW IF EXISTS ${hiveconf:TEMP_TABLE1};
CREATE VIEW ${hiveconf:TEMP_TABLE1} AS
SELECT
    i_item_sk item_id,
    sum(sr_return_quantity) sr_item_qty
FROM 
    store_returns sr
    INNER JOIN item i ON sr.sr_item_sk = i.i_item_sk
    INNER JOIN date_dim d ON sr.sr_returned_date_sk = d.d_date_sk
    INNER JOIN (	
					SELECT d1.d_date_sk FROM date_dim d1 
					LEFT SEMI JOIN date_dim d2 ON (d1.d_week_seq = d2.d_week_seq AND d2.d_date IN ('1998-01-02','1998-10-15','1998-11-10')) 
				) d1 ON sr.sr_returned_date_sk = d1.d_date_sk
GROUP BY i_item_sk
HAVING sum(sr_return_quantity) > 0;


-----Web returns in daterange q19_tmp_date2 ------------------------------------------------------
DROP VIEW IF EXISTS ${hiveconf:TEMP_TABLE2};
--make ${hiveconf:TEMP_TABLE2}
CREATE VIEW ${hiveconf:TEMP_TABLE2} AS
SELECT
    i_item_sk item_id,
    sum(wr_return_quantity) wr_item_qty
FROM 
    web_returns wr
    INNER JOIN item i ON wr.wr_item_sk = i.i_item_sk
    INNER JOIN date_dim d ON wr.wr_returned_date_sk = d.d_date_sk
    INNER JOIN (
					SELECT d1.d_date_sk FROM date_dim d1 
					LEFT SEMI JOIN date_dim d2 ON (d1.d_week_seq = d2.d_week_seq AND d2.d_date IN ('2001-03-10' ,'2001-08-04' ,'2001-11-14')) 	
				) d2 ON wr.wr_returned_date_sk = d2.d_date_sk
GROUP BY i_item_sk
HAVING sum(wr_return_quantity) > 0;

----return items -------------------------------------------------------------------
DROP VIEW IF EXISTS ${hiveconf:TEMP_TABLE3};
--make ${hiveconf:TEMP_TABLE3}
CREATE VIEW ${hiveconf:TEMP_TABLE3} AS
	SELECT 
		st.item_id item,
		sr_item_qty,
		100.0*sr_item_qty/(sr_item_qty+wr_item_qty)/2.0 sr_dev,
		wr_item_qty,
		100.0*wr_item_qty/(sr_item_qty+wr_item_qty)/2.0 wr_dev,
		(sr_item_qty+wr_item_qty)/2.0 average
	FROM ${hiveconf:TEMP_TABLE1} st
	INNER JOIN ${hiveconf:TEMP_TABLE2} wt ON st.item_id = wt.item_id
	ORDER BY average desc
	LIMIT 100;

---Sentiment analysis and Result----------------------------------------------------------------
--- we can reuse the  sentiment analysis helper class from q10
ADD JAR ${env:BIG_BENCH_QUERIES_DIR}/Resources/bigbenchqueriesmr.jar;
CREATE TEMPORARY FUNCTION extract_sentiment AS 'de.bankmark.bigbench.queries.q10.SentimentUDF';


--Result  returned items with negative sentiment --------------------------------------------		
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;	

--CREATE RESULT TABLE. Store query result externally in output_dir/qXXresult/
DROP TABLE IF EXISTS ${hiveconf:RESULT_TABLE};
CREATE TABLE ${hiveconf:RESULT_TABLE}
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'
STORED AS ${env:BIG_BENCH_hive_default_fileformat_result_table} LOCATION '${hiveconf:RESULT_DIR}' 
AS
---- the real query --------------
SELECT * 
FROM
(
	SELECT extract_sentiment(pr.pr_item_sk, pr.pr_review_content)  
	AS(	pr_item_sk, 
		review_sentence, 
		sentiment, 
		sentiment_word
	)  
	FROM product_reviews  pr
	LEFT SEMI JOIN ${hiveconf:TEMP_TABLE3} ri ON pr.pr_item_sk = ri.item
) q19_tmp_sentiment
WHERE q19_tmp_sentiment.sentiment = 'NEG ';

	
--- cleanup---------------------------------------------------------------------------


DROP VIEW IF EXISTS ${hiveconf:TEMP_TABLE1};
DROP VIEW IF EXISTS ${hiveconf:TEMP_TABLE2};
DROP VIEW IF EXISTS ${hiveconf:TEMP_TABLE3};
