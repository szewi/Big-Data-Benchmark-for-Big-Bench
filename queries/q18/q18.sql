--Identify the stores with flat or declining sales in 3 consecutive months,
--check if there are any negative reviews regarding these stores available online.

-- Resources
ADD JAR ${env:BIG_BENCH_QUERIES_DIR}/Resources/opennlp-maxent-3.0.3.jar;
ADD JAR ${env:BIG_BENCH_QUERIES_DIR}/Resources/opennlp-tools-1.5.3.jar;
ADD JAR ${env:BIG_BENCH_QUERIES_DIR}/Resources/bigbenchqueriesmr.jar;
CREATE TEMPORARY FUNCTION extract_NegSentiment AS 'de.bankmark.bigbench.queries.q18.NegativeSentimentUDF';


DROP TABLE IF EXISTS ${hiveconf:TEMP_TABLE};
CREATE TABLE ${hiveconf:TEMP_TABLE}
AS
SELECT 	 
	 s_store_name
	,pr_review_date
	,pr_review_content
FROM 
(
 --select store_name for stores with flat or declining sales in 3 consecutive months.
	SELECT s_store_name
	FROM  store s
	JOIN 
	(   
	    -- linear regression part
		SELECT 
			temp.cat AS cat,
			--sum(temp.x)as sumX,
			--sum(temp.y)as sumY,
			--sum(temp.xy)as sumXY,
			--sum(temp.xx)as sumXSquared,
			--count(temp.x) as N,
			--N * sumXY - sumX * sumY AS numerator,
			--N * sumXSquared - sumX*sumX AS denom
			--numerator / denom as slope,
			--(sumY - slope * sumX) / N as intercept

			--(count(temp.x) * sum(temp.xy) - sum(temp.x) * sum(temp.y)) AS numerator,
			--(count(temp.x) * sum(temp.xx) - sum(temp.x)*sum(temp.x)) AS denom

			--numerator / denom as slope,
			--(sumY - slope * sumX) / N as intercept
			((count(temp.x) * sum(temp.xy) - sum(temp.x) * sum(temp.y)) / (count(temp.x) * sum(temp.xx) - sum(temp.x)*sum(temp.x)) ) as slope,
			(sum(temp.y) - ((count(temp.x) * sum(temp.xy) - sum(temp.x) * sum(temp.y)) / (count(temp.x) * sum(temp.xx) - sum(temp.x)*sum(temp.x)) ) * sum(temp.x)) / count(temp.x) as intercept
		FROM
		(
			SELECT 
				s.ss_store_sk 	 AS cat,
				s.ss_sold_date_sk  AS x,
				sum(s.ss_net_paid) AS y,
				s.ss_sold_date_sk*sum(s.ss_net_paid) AS xy,
				s.ss_sold_date_sk*s.ss_sold_date_sk AS xx
			FROM store_sales s
			--select date range
			LEFT SEMI JOIN (	
					SELECT d_date_sk 
					FROM  date_dim d
					WHERE d.d_date >= '${hiveconf:q18_startDate}'
					AND   d.d_date <= '${hiveconf:q18_endDate}'
				) dd ON ( s.ss_sold_date_sk=dd.d_date_sk ) 
			where s.ss_store_sk <=18
			GROUP BY s.ss_store_sk, s.ss_sold_date_sk
		)temp
		group by temp.cat	
	)c on s.s_store_sk = c.cat
	WHERE  c.slope < 0 
) tmp
JOIN   product_reviews pr on (true)
where instr(pr.pr_review_content,  tmp.s_store_name) >0
;




--Result  --------------------------------------------------------------------		
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;	
--Prepare result storage
DROP TABLE IF EXISTS ${hiveconf:RESULT_TABLE};
CREATE TABLE ${hiveconf:RESULT_TABLE}
	ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'
	STORED AS ${env:BIG_BENCH_hive_default_fileformat_result_table} LOCATION '${hiveconf:RESULT_DIR}'
AS
-- the real query
SELECT 	extract_NegSentiment( s_store_name, pr_review_date, pr_review_content) AS ( s_store_name, review_date, review_sentence, sentiment, sentiment_word )	
--select product_reviews containing the name of a store. Consider only stores with flat or declining sales in 3 consecutive months.
FROM  ${hiveconf:TEMP_TABLE}
;


-- Cleanup ----------------------------------------
DROP TABLE IF EXISTS ${hiveconf:TEMP_TABLE};

