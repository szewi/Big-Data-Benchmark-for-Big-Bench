--Build text classifier for online review sentiment classification (Positive,
--Negative, Neutral), using 60% of available reviews for training and the remaining
--40% for testing. Display classifier accuracy on testing data.

-- Query parameters


-- Resources
--ADD FILE ${env:BIG_BENCH_QUERIES_DIR}/q28/mapper_q28.py;

--Result 1 Training table for mahout--------------------------------------------------------------------		
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;
DROP TABLE IF EXISTS ${hiveconf:TEMP_TABLE1};
CREATE TABLE ${hiveconf:TEMP_TABLE1} 
	ROW FORMAT DELIMITED
	FIELDS TERMINATED BY '\t'
	LINES TERMINATED BY '\n'
	STORED AS TEXTFILE
	--STORED AS SEQUENCEFILE 
	LOCATION '${hiveconf:TEMP_DIR1}'
AS
SELECT  pr_review_sk,
	CASE pr_review_rating
		when 1 then 'NEG'
		when 2 then 'NEG'
		when 3 then 'NEU'
		when 4 then 'POS'
		when 5 then 'POS'
    	END AS pr_rating,
    pr_review_content
FROM product_reviews
WHERE pmod(pr_review_sk, 5) IN (1,2,3)
--limit 10000
;


--Result 2 Testing table for mahout --------------------------------------------------------------------		
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;
DROP TABLE IF EXISTS ${hiveconf:TEMP_TABLE2};
CREATE TABLE ${hiveconf:TEMP_TABLE2}
	ROW FORMAT DELIMITED
	FIELDS TERMINATED BY '\t'
	LINES TERMINATED BY '\n'
	STORED AS TEXTFILE
	--STORED AS SEQUENCEFILE 
	LOCATION '${hiveconf:TEMP_DIR2}'
AS
SELECT  pr_review_sk,
	CASE pr_review_rating
		when 1 then 'NEG'
		when 2 then 'NEG'
		when 3 then 'NEU'
		when 4 then 'POS'
		when 5 then 'POS'
    	END AS pr_rating,
    pr_review_content
FROM product_reviews
WHERE pmod(pr_review_sk, 5) in (0,4)
--limit 10000
;
