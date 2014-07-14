--Find the categories with flat or declining sales for in store purchases
--during a given year for a given store.

-- Resources

--Result  --------------------------------------------------------------------		
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;

--CREATE RESULT TABLE. Store query result externally in output_dir/qXXresult/
DROP TABLE IF EXISTS ${hiveconf:RESULT_TABLE};
CREATE TABLE ${hiveconf:RESULT_TABLE}
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'
STORED AS ${env:BIG_BENCH_hive_default_fileformat_result_table} LOCATION '${hiveconf:RESULT_DIR}' 
AS
SELECT * 
FROM
(
	SELECT 
		temp.cat,
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
			i.i_category_id 	AS cat, -- ranges from 1 to 10
			s.ss_sold_date_sk 	AS x,
			sum(s.ss_net_paid)	AS y,
			s.ss_sold_date_sk*sum(s.ss_net_paid) AS xy,
			s.ss_sold_date_sk*s.ss_sold_date_sk AS xx
		FROM    store_sales s
		-- select date range 
		LEFT SEMI JOIN 
		(	
			SELECT d_date_sk 
			FROM  date_dim d
			WHERE d.d_date >= '${hiveconf:q15_startDate}'
			AND   d.d_date <= '${hiveconf:q15_endDate}'
		) dd ON ( s.ss_sold_date_sk=dd.d_date_sk ) 
		INNER JOIN item i ON s.ss_item_sk = i.i_item_sk 
		WHERE i.i_category_id IS NOT NULL
		  AND s.ss_store_sk = ${hiveconf:q15_store_sk} -- for a given store ranges from 1 to 12
		GROUP BY i.i_category_id, s.ss_sold_date_sk

	)temp
	group by temp.cat
)regression
where slope < 0
;
