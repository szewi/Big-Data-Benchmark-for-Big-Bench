-------- Q01 -----------
--category_ids:
--1 Home & Kitchen
--2 Music
--3 Books
--4 Clothing & Accessories
--5 Electronics
--6 Tools & Home Improvement
--7 Toys & Games
--8 Movies & TV
--9 Sports & Outdoors
set q01_i_category_id_IN=1, 2 ,3;
-- sf1 -> 11 stores, 90k sales in 820k lines
set q01_ss_store_sk_IN=10, 20, 33, 40, 50;
set q01_COUNT_pid1_greater=49;

-------- Q02 -----------
-- q02_pid1_IN=<pid>, <pid>, ..
--pid == item_sk
--sf 1 item count: 17999
set q02_pid1_IN=1415;
set q02_NPATH_ITEM_SET_MAX=500;
set q02_limit=300000;


-------- Q03 -----------
-- no results-check reduce script
set q03_days_before_purchase=10;
set q03_purchased_item_IN=16891;


-------- Q04 -----------
--no params

-------- Q05 -----------
set q05_i_category='Books';
set q05_cd_education_status_IN='Advanced Degree', 'College', '4 yr Degree', '2 yr Degree';
set q05_cd_gender='M';


-------- Q06 -----------
SET q06_LIMIT=100;
SET q06_YEAR=1999;

-------- Q07 -----------
SET q07_HIGHER_PRICE_RATIO=1.2;
SET q07_YEAR=2002;
SET q07_MONTH=7;
SET q07_HAVING_COUNT_GE=10;
SET q07_LIMIT=100;

-------- Q08 -----------
set q08_category=review;
set q08_startDate=1999-09-02;
set q08_endDate=2000-09-02;


-------- Q09 -----------
-- Query parameters
set q09_year=1998;

set q09_part1_ca_country=United States;
set q09_part1_ca_state_IN='KY', 'GA', 'NM';
set q09_part1_net_profit_min=0;
set q09_part1_net_profit_max=2000;
set q09_part1_education_status=4 yr Degree;
set q09_part1_marital_status=M;
set q09_part1_sales_price_min=100;
set q09_part1_sales_price_max=150;

set q09_part2_ca_country=United States;
set q09_part2_ca_state_IN='MT', 'OR', 'IN';
set q09_part2_net_profit_min=150;
set q09_part2_net_profit_max=3000;
set q09_part2_education_status=4 yr Degree;
set q09_part2_marital_status=M;
set q09_part2_sales_price_min=50;
set q09_part2_sales_price_max=200;

set q09_part3_ca_country=United States;
set q09_part3_ca_state_IN='WI', 'MO', 'WV';
set q09_part3_net_profit_min=50;
set q09_part3_net_profit_max=25000;
set q09_part3_education_status=4 yr Degree;
set q09_part3_marital_status=M;
set q09_part3_sales_price_min=150;
set q09_part3_sales_price_max=200;

-------- Q10 -----------
--no params

-------- Q11 -----------
set q11_startDate=2003-01-02;
set q11_endDate=2003-01-01;
	
	
-------- Q12 -----------
set q12_startDate=1999-09-02;
set q12_endDate1=1999-09-03;
set q12_endDate2=1999-09-06;
set q12_i_category_IN='Books', 'Electronics';

-------- Q13 -----------
set q13_Year=1999;
set q13_limit=100;

-------- Q14 -----------
set q14_dependents=5;
set q14_morning_startHour=7;
set q14_morning_endHour=8;
set q14_evening_startHour=19;
set q14_evening_endHour=20;
set q14_content_len_min=5000;
set q14_content_len_max=5200;

-------- Q15 -----------
set q15_startDate=1999-09-02;
set q15_endDate=2000-09-02;
set q15_store_sk=10;


-------- Q16 -----------
set q16_date=1998-03-16;

-------- Q17 -----------
set q17_gmt_offset=-7;
set q17_year=2001;
set q17_month=12;
set q17_i_category_IN='Jewelry';

-------- Q18 -----------
set q18_startDate=1999-09-02;
set q18_endDate=1999-05-02;

-------- Q19 -----------
set q19_storeReturns_date_IN='1998-01-02','1998-10-15','1998-11-10';
set q19_webReturns_date_IN='2001-03-10' ,'2001-08-04' ,'2001-11-14';
set q19_store_return_limit=100;

-------- Q20 -----------
--no params

-------- Q21 -----------
set q21_year=1998;
set q21_month=4;

-------- Q22 -----------
set q22_date=2000-05-08;
set q22_i_current_price_min=0.98;
set q22_i_current_price_max=1.5;

-------- Q23 -----------
set q23_year=1998;
set q23_month=1;
set q23_coeficient=1.5;

-------- Q24 -----------
set q24_i_item_sk_IN=7, 17;

-------- Q25 -----------
set q25_date=2002-01-02;

-------- Q26 -----------
set q26_i_category_IN='Books';
set q26_count_ss_item_sk=5;

-------- Q27 -----------
set q27_pr_item_sk=10653;

-------- Q28 -----------
--no params

-------- Q29 -----------
--no params

-------- Q30 -----------
--no params

