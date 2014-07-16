set hdfsDataPath=${env:BIG_BENCH_HDFS_ABSOLUTE_DATA_DIR};
set fieldDelimiter=|;

set customerTableName=customer;
set customerAddressTableName=customer_address;
set customerDemographicsTableName=customer_demographics;
set dateTableName=date_dim;
set householdDemographicsTableName=household_demographics;
set incomeTableName=income_band;
set itemTableName=item;
set promotionTableName=promotion;
set reasonTableName=reason;
set shipModeTableName=ship_mode;
set storeTableName=store;
set timeTableName=time_dim;
set warehouseTableName=warehouse;
set webSiteTableName=web_site;
set webPageTableName=web_page;
set inventoryTableName=inventory;
set storeSalesTableName=store_sales;
set storeReturnsTableName=store_returns;
set webSalesTableName=web_sales;
set webReturnsTableName=web_returns;

set marketPricesTableName=item_marketprices;
set clickstreamsTableName=web_clickstreams;
set reviewsTableName=product_reviews;

!echo drop temp tables;
DROP TABLE IF EXISTS ${hiveconf:customerTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:customerAddressTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:customerDemographicsTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:dateTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:householdDemographicsTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:incomeTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:itemTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:promotionTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:reasonTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:shipModeTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:storeTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:timeTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:warehouseTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:webSiteTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:webPageTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:inventoryTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:storeSalesTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:storeReturnsTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:webSalesTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:webReturnsTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:marketPricesTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:clickstreamsTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:reviewsTableName}_textFormat;


!echo Create table: ${hiveconf:customerDemographicsTableName}_textFormat;
CREATE EXTERNAL TABLE ${hiveconf:customerDemographicsTableName}_textFormat
  ( cd_demo_sk                bigint                ----not null
  , cd_gender                 string
  , cd_marital_status         string
  , cd_education_status       string
  , cd_purchase_estimate      int
  , cd_credit_rating          string
  , cd_dep_count              int
  , cd_dep_employed_count     int
  , cd_dep_college_count      int

  )
  ROW FORMAT DELIMITED FIELDS TERMINATED BY '${hiveconf:fieldDelimiter}'
  STORED AS TEXTFILE LOCATION '${hiveconf:hdfsDataPath}/${hiveconf:customerDemographicsTableName}'
;


!echo Create table: ${hiveconf:dateTableName}_textFormat;
CREATE EXTERNAL TABLE ${hiveconf:dateTableName}_textFormat
  ( d_date_sk                 bigint              --not null
  , d_date_id                 string              --not null
  , d_date                    string
  , d_month_seq               int
  , d_week_seq                int
  , d_quarter_seq             int
  , d_year                    int
  , d_dow                     int
  , d_moy                     int
  , d_dom                     int
  , d_qoy                     int
  , d_fy_year                 int
  , d_fy_quarter_seq          int
  , d_fy_week_seq             int
  , d_day_name                string
  , d_quarter_name            string
  , d_holiday                 string
  , d_weekend                 string
  , d_following_holiday       string
  , d_first_dom               int
  , d_last_dom                int
  , d_same_day_ly             int
  , d_same_day_lq             int
  , d_current_day             string
  , d_current_week            string
  , d_current_month           string
  , d_current_quarter         string
  , d_current_year            string
  )
  ROW FORMAT DELIMITED FIELDS TERMINATED BY '${hiveconf:fieldDelimiter}'
  STORED AS TEXTFILE LOCATION '${hiveconf:hdfsDataPath}/${hiveconf:dateTableName}'
;


!echo Create table: ${hiveconf:warehouseTableName}_textFormat;
CREATE EXTERNAL TABLE ${hiveconf:warehouseTableName}_textFormat
  ( w_warehouse_sk            bigint              --not null
  , w_warehouse_id            string              --not null
  , w_warehouse_name          string
  , w_warehouse_sq_ft         int
  , w_street_number           string
  , w_street_name             string
  , w_street_type             string
  , w_suite_number            string
  , w_city                    string
  , w_county                  string
  , w_state                   string
  , w_zip                     string
  , w_country                 string
  , w_gmt_offset              double
  )
  ROW FORMAT DELIMITED FIELDS TERMINATED BY '${hiveconf:fieldDelimiter}'
  STORED AS TEXTFILE LOCATION '${hiveconf:hdfsDataPath}/${hiveconf:warehouseTableName}'
;


!echo Create table: ${hiveconf:shipModeTableName}_textFormat;
CREATE EXTERNAL TABLE ${hiveconf:shipModeTableName}_textFormat
  ( sm_ship_mode_sk           bigint              --not null
  , sm_ship_mode_id           string              --not null
  , sm_type                   string
  , sm_code                   string
  , sm_carrier                string
  , sm_contract               string

  )
  ROW FORMAT DELIMITED FIELDS TERMINATED BY '${hiveconf:fieldDelimiter}'
  STORED AS TEXTFILE LOCATION '${hiveconf:hdfsDataPath}/${hiveconf:shipModeTableName}'
;


!echo Create table: ${hiveconf:timeTableName}_textFormat;
CREATE EXTERNAL TABLE ${hiveconf:timeTableName}_textFormat
  ( t_time_sk                 bigint              --not null
  , t_time_id                 string              --not null
  , t_time                    int
  , t_hour                    int
  , t_minute                  int
  , t_second                  int
  , t_am_pm                   string
  , t_shift                   string
  , t_sub_shift               string
  , t_meal_time               string
  )
  ROW FORMAT DELIMITED FIELDS TERMINATED BY '${hiveconf:fieldDelimiter}'
  STORED AS TEXTFILE LOCATION '${hiveconf:hdfsDataPath}/${hiveconf:timeTableName}'
;


!echo Create table: ${hiveconf:reasonTableName}_textFormat;
CREATE EXTERNAL TABLE ${hiveconf:reasonTableName}_textFormat
  ( r_reason_sk               bigint              --not null
  , r_reason_id               string              --not null
  , r_reason_desc             string
  )
  ROW FORMAT DELIMITED FIELDS TERMINATED BY '${hiveconf:fieldDelimiter}'
  STORED AS TEXTFILE LOCATION '${hiveconf:hdfsDataPath}/${hiveconf:reasonTableName}'
;

!echo Create table: ${hiveconf:incomeTableName}_textFormat;
CREATE EXTERNAL TABLE ${hiveconf:incomeTableName}_textFormat
  ( ib_income_band_sk         bigint              --not null
  , ib_lower_bound            int
  , ib_upper_bound            int
  )
  ROW FORMAT DELIMITED FIELDS TERMINATED BY '${hiveconf:fieldDelimiter}'
  STORED AS TEXTFILE LOCATION '${hiveconf:hdfsDataPath}/${hiveconf:incomeTableName}'
;

!echo Create table: ${hiveconf:storeTableName}_textFormat;
CREATE EXTERNAL TABLE ${hiveconf:storeTableName}_textFormat
  ( s_store_sk                bigint              --not null
  , s_store_id                string              --not null
  , s_rec_start_date          string
  , s_rec_end_date            string
  , s_closed_date_sk          bigint
  , s_store_name              string
  , s_number_employees        int
  , s_floor_space             int
  , s_hours                   string
  , s_manager                 string
  , s_market_id               int
  , s_geography_class         string
  , s_market_desc             string
  , s_market_manager          string
  , s_division_id             int
  , s_division_name           string
  , s_company_id              int
  , s_company_name            string
  , s_street_number           string
  , s_street_name             string
  , s_street_type             string
  , s_suite_number            string
  , s_city                    string
  , s_county                  string
  , s_state                   string
  , s_zip                     string
  , s_country                 string
  , s_gmt_offset              double
  , s_tax_precentage          double
  )
  ROW FORMAT DELIMITED FIELDS TERMINATED BY '${hiveconf:fieldDelimiter}'
  STORED AS TEXTFILE LOCATION '${hiveconf:hdfsDataPath}/${hiveconf:storeTableName}'
;

!echo Create table: ${hiveconf:webSiteTableName}_textFormat;
CREATE EXTERNAL TABLE ${hiveconf:webSiteTableName}_textFormat
  ( web_site_sk               bigint              --not null
  , web_site_id               string              --not null
  , web_rec_start_date        string
  , web_rec_end_date          string
  , web_name                  string
  , web_open_date_sk          bigint
  , web_close_date_sk         bigint
  , web_class                 string
  , web_manager               string
  , web_mkt_id                int
  , web_mkt_class             string
  , web_mkt_desc              string
  , web_market_manager        string
  , web_company_id            int
  , web_company_name          string
  , web_street_number         string
  , web_street_name           string
  , web_street_type           string
  , web_suite_number          string
  , web_city                  string
  , web_county                string
  , web_state                 string
  , web_zip                   string
  , web_country               string
  , web_gmt_offset            double
  , web_tax_percentage        double
  )
  ROW FORMAT DELIMITED FIELDS TERMINATED BY '${hiveconf:fieldDelimiter}'
  STORED AS TEXTFILE LOCATION '${hiveconf:hdfsDataPath}/${hiveconf:webSiteTableName}'
;


!echo Create table: ${hiveconf:householdDemographicsTableName}_textFormat;
CREATE EXTERNAL TABLE ${hiveconf:householdDemographicsTableName}_textFormat
  ( hd_demo_sk                bigint                --not null
  , hd_income_band_sk         bigint
  , hd_buy_potential          string
  , hd_dep_count              int
  , hd_vehicle_count          int
  )
  ROW FORMAT DELIMITED FIELDS TERMINATED BY '${hiveconf:fieldDelimiter}'
  STORED AS TEXTFILE LOCATION '${hiveconf:hdfsDataPath}/${hiveconf:householdDemographicsTableName}'
;


!echo Create table: ${hiveconf:webPageTableName}_textFormat;
CREATE EXTERNAL TABLE ${hiveconf:webPageTableName}_textFormat
  ( wp_web_page_sk            bigint              --not null
  , wp_web_page_id            string              --not null
  , wp_rec_start_date         string
  , wp_rec_end_date           string
  , wp_creation_date_sk       bigint
  , wp_access_date_sk         bigint
  , wp_autogen_flag           string
  , wp_customer_sk            bigint
  , wp_url                    string
  , wp_type                   string
  , wp_char_count             int
  , wp_link_count             int
  , wp_image_count            int
  , wp_max_ad_count           int
  )
  ROW FORMAT DELIMITED FIELDS TERMINATED BY '${hiveconf:fieldDelimiter}'
  STORED AS TEXTFILE LOCATION '${hiveconf:hdfsDataPath}/${hiveconf:webPageTableName}'
;


!echo Create table: ${hiveconf:promotionTableName}_textFormat;
CREATE EXTERNAL TABLE ${hiveconf:promotionTableName}_textFormat
  ( p_promo_sk                bigint              --not null
  , p_promo_id                string              --not null
  , p_start_date_sk           bigint
  , p_end_date_sk             bigint
  , p_item_sk                 bigint
  , p_cost                    double
  , p_response_target         int
  , p_promo_name              string
  , p_channel_dmail           string
  , p_channel_email           string
  , p_channel_catalog         string
  , p_channel_tv              string
  , p_channel_radio           string
  , p_channel_press           string
  , p_channel_event           string
  , p_channel_demo            string
  , p_channel_details         string
  , p_purpose                 string
  , p_discount_active         string
  )
  ROW FORMAT DELIMITED FIELDS TERMINATED BY '${hiveconf:fieldDelimiter}'
  STORED AS TEXTFILE LOCATION '${hiveconf:hdfsDataPath}/${hiveconf:promotionTableName}'
;


!echo Create table: ${hiveconf:customerTableName}_textFormat;
CREATE EXTERNAL TABLE ${hiveconf:customerTableName}_textFormat
  ( c_customer_sk             bigint              --not null
  , c_customer_id             string              --not null
  , c_current_cdemo_sk        bigint
  , c_current_hdemo_sk        bigint
  , c_current_addr_sk         bigint
  , c_first_shipto_date_sk    bigint
  , c_first_sales_date_sk     bigint
  , c_salutation              string
  , c_first_name              string
  , c_last_name               string
  , c_preferred_cust_flag     string
  , c_birth_day               int
  , c_birth_month             int
  , c_birth_year              int
  , c_birth_country           string
  , c_login                   string
  , c_email_address           string
  , c_last_review_date        string
  )
  ROW FORMAT DELIMITED FIELDS TERMINATED BY '${hiveconf:fieldDelimiter}'
  STORED AS TEXTFILE LOCATION '${hiveconf:hdfsDataPath}/${hiveconf:customerTableName}'
  ;


!echo Create table: ${hiveconf:customerAddressTableName}_textFormat;
CREATE EXTERNAL TABLE ${hiveconf:customerAddressTableName}_textFormat
  ( ca_address_sk             bigint              --not null
  , ca_address_id             string              --not null
  , ca_street_number          string
  , ca_street_name            string
  , ca_street_type            string
  , ca_suite_number           string
  , ca_city                   string
  , ca_county                 string
  , ca_state                  string
  , ca_zip                    string
  , ca_country                string
  , ca_gmt_offset             double
  , ca_location_type          string
  )
  ROW FORMAT DELIMITED FIELDS TERMINATED BY '${hiveconf:fieldDelimiter}'
  STORED AS TEXTFILE LOCATION '${hiveconf:hdfsDataPath}/${hiveconf:customerAddressTableName}'
  ;


!echo Create table: ${hiveconf:inventoryTableName}_textFormat;
CREATE EXTERNAL TABLE ${hiveconf:inventoryTableName}_textFormat
  ( inv_date_sk               bigint                --not null
  , inv_item_sk               bigint                --not null
  , inv_warehouse_sk          bigint                --not null
  , inv_quantity_on_hand      int
  )
  ROW FORMAT DELIMITED FIELDS TERMINATED BY '${hiveconf:fieldDelimiter}'
  STORED AS TEXTFILE LOCATION '${hiveconf:hdfsDataPath}/${hiveconf:inventoryTableName}'
;


!echo Create table: ${hiveconf:itemTableName}_textFormat;
CREATE EXTERNAL TABLE ${hiveconf:itemTableName}_textFormat
  ( i_item_sk                 bigint              --not null
  , i_item_id                 string              --not null
  , i_rec_start_date          string
  , i_rec_end_date            string
  , i_item_desc               string
  , i_current_price           double
  , i_wholesale_cost          double
  , i_brand_id                int
  , i_brand                   string
  , i_class_id                int
  , i_class                   string
  , i_category_id             int
  , i_category                string
  , i_manufact_id             int
  , i_manufact                string
  , i_size                    string
  , i_formulation             string
  , i_color                   string
  , i_units                   string
  , i_container               string
  , i_manager_id              int
  , i_product_name            string
  )
  ROW FORMAT DELIMITED FIELDS TERMINATED BY '${hiveconf:fieldDelimiter}'
  STORED AS TEXTFILE LOCATION '${hiveconf:hdfsDataPath}/${hiveconf:itemTableName}'
;


!echo Create table: ${hiveconf:storeSalesTableName}_textFormat;
CREATE EXTERNAL TABLE ${hiveconf:storeSalesTableName}_textFormat
  ( ss_sold_date_sk           bigint
  , ss_sold_time_sk           bigint
  , ss_item_sk                bigint                --not null
  , ss_customer_sk            bigint
  , ss_cdemo_sk               bigint
  , ss_hdemo_sk               bigint
  , ss_addr_sk                bigint
  , ss_store_sk               bigint
  , ss_promo_sk               bigint
  , ss_ticket_number          bigint                --not null
  , ss_quantity               int
  , ss_wholesale_cost         double
  , ss_list_price             double
  , ss_sales_price            double
  , ss_ext_discount_amt       double
  , ss_ext_sales_price        double
  , ss_ext_wholesale_cost     double
  , ss_ext_list_price         double
  , ss_ext_tax                double
  , ss_coupon_amt             double
  , ss_net_paid               double
  , ss_net_paid_inc_tax       double
  , ss_net_profit             double
  )
  ROW FORMAT DELIMITED FIELDS TERMINATED BY '${hiveconf:fieldDelimiter}'
  STORED AS TEXTFILE LOCATION '${hiveconf:hdfsDataPath}/${hiveconf:storeSalesTableName}'
;


!echo Create table: ${hiveconf:storeReturnsTableName}_textFormat;
CREATE EXTERNAL TABLE ${hiveconf:storeReturnsTableName}_textFormat
  ( sr_returned_date_sk       bigint
  , sr_return_time_sk         bigint
  , sr_item_sk                bigint                --not null
  , sr_customer_sk            bigint
  , sr_cdemo_sk               bigint
  , sr_hdemo_sk               bigint
  , sr_addr_sk                bigint
  , sr_store_sk               bigint
  , sr_reason_sk              bigint
  , sr_ticket_number          bigint                --not null
  , sr_return_quantity        int
  , sr_return_amt             double
  , sr_return_tax             double
  , sr_return_amt_inc_tax     double
  , sr_fee                    double
  , sr_return_ship_cost       double
  , sr_refunded_cash          double
  , sr_reversed_charge        double
  , sr_store_credit           double
  , sr_net_loss               double
  )
  ROW FORMAT DELIMITED FIELDS TERMINATED BY '${hiveconf:fieldDelimiter}'
  STORED AS TEXTFILE LOCATION '${hiveconf:hdfsDataPath}/${hiveconf:storeReturnsTableName}'
;


!echo Create table: ${hiveconf:webSalesTableName}_textFormat;
CREATE EXTERNAL TABLE ${hiveconf:webSalesTableName}_textFormat
  ( ws_sold_date_sk           bigint
  , ws_sold_time_sk           bigint
  , ws_ship_date_sk           bigint
  , ws_item_sk                bigint                --not null
  , ws_bill_customer_sk       bigint
  , ws_bill_cdemo_sk          bigint
  , ws_bill_hdemo_sk          bigint
  , ws_bill_addr_sk           bigint
  , ws_ship_customer_sk       bigint
  , ws_ship_cdemo_sk          bigint
  , ws_ship_hdemo_sk          bigint
  , ws_ship_addr_sk           bigint
  , ws_web_page_sk            bigint
  , ws_web_site_sk            bigint
  , ws_ship_mode_sk           bigint
  , ws_warehouse_sk           bigint
  , ws_promo_sk               bigint
  , ws_order_number           bigint                --not null
  , ws_quantity               int
  , ws_wholesale_cost         double
  , ws_list_price             double
  , ws_sales_price            double
  , ws_ext_discount_amt       double
  , ws_ext_sales_price        double
  , ws_ext_wholesale_cost     double
  , ws_ext_list_price         double
  , ws_ext_tax                double
  , ws_coupon_amt             double
  , ws_ext_ship_cost          double
  , ws_net_paid               double
  , ws_net_paid_inc_tax       double
  , ws_net_paid_inc_ship      double
  , ws_net_paid_inc_ship_tax  double
  , ws_net_profit             double
  )
  ROW FORMAT DELIMITED FIELDS TERMINATED BY '${hiveconf:fieldDelimiter}'
  STORED AS TEXTFILE LOCATION '${hiveconf:hdfsDataPath}/${hiveconf:webSalesTableName}'
;


!echo Create table: ${hiveconf:webReturnsTableName}_textFormat;
CREATE EXTERNAL TABLE ${hiveconf:webReturnsTableName}_textFormat
  ( wr_returned_date_sk       bigint 
  , wr_returned_time_sk       bigint
  , wr_item_sk                bigint                --not null
  , wr_refunded_customer_sk   bigint
  , wr_refunded_cdemo_sk      bigint
  , wr_refunded_hdemo_sk      bigint
  , wr_refunded_addr_sk       bigint
  , wr_returning_customer_sk  bigint
  , wr_returning_cdemo_sk     bigint
  , wr_returning_hdemo_sk     bigint
  , wr_returning_addr_sk      bigint
  , wr_web_page_sk            bigint
  , wr_reason_sk              bigint
  , wr_order_number           bigint                --not null
  , wr_return_quantity        int
  , wr_return_amt             double
  , wr_return_tax             double
  , wr_return_amt_inc_tax     double
  , wr_fee                    double
  , wr_return_ship_cost       double
  , wr_refunded_cash          double
  , wr_reversed_charge        double
  , wr_account_credit         double
  , wr_net_loss               double
  )
  ROW FORMAT DELIMITED FIELDS TERMINATED BY '${hiveconf:fieldDelimiter}'
  STORED AS TEXTFILE LOCATION '${hiveconf:hdfsDataPath}/${hiveconf:webReturnsTableName}'
;


!echo Create table: ${hiveconf:marketPricesTableName}_textFormat;
CREATE EXTERNAL TABLE ${hiveconf:marketPricesTableName}_textFormat
  ( imp_sk                  bigint                --not null
  , imp_item_sk             bigint                --not null
  , imp_competitor          string
  , imp_competitor_price    double
  , imp_start_date          bigint
  , imp_end_date            bigint

  )
  ROW FORMAT DELIMITED FIELDS TERMINATED BY '${hiveconf:fieldDelimiter}'
  STORED AS TEXTFILE LOCATION '${hiveconf:hdfsDataPath}/${hiveconf:marketPricesTableName}'
  ;


!echo Create table: ${hiveconf:clickstreamsTableName}_textFormat;
CREATE EXTERNAL TABLE ${hiveconf:clickstreamsTableName}_textFormat
(   wcs_click_date_sk       bigint
  , wcs_click_time_sk       bigint
  , wcs_sales_sk            bigint
  , wcs_item_sk             bigint
  , wcs_web_page_sk         bigint
  , wcs_user_sk             bigint

  )
  ROW FORMAT DELIMITED FIELDS TERMINATED BY '${hiveconf:fieldDelimiter}'
  STORED AS TEXTFILE LOCATION '${hiveconf:hdfsDataPath}/${hiveconf:clickstreamsTableName}'
  ;


!echo Create table: ${hiveconf:reviewsTableName}_textFormat;
CREATE EXTERNAL TABLE ${hiveconf:reviewsTableName}_textFormat
(   pr_review_sk            bigint              --not null
  , pr_review_date          string
  , pr_review_time          string 
  , pr_review_rating        int                 --not null
  , pr_item_sk              bigint              --not null
  , pr_user_sk              bigint
  , pr_order_sk             bigint
  , pr_review_content       string --not null
  )
  ROW FORMAT DELIMITED FIELDS TERMINATED BY '${hiveconf:fieldDelimiter}'
  STORED AS TEXTFILE LOCATION '${hiveconf:hdfsDataPath}/${hiveconf:reviewsTableName}'
  ;


!echo Load text data into ORC table: ${hiveconf:customerDemographicsTableName};
CREATE TABLE IF NOT EXISTS ${hiveconf:customerDemographicsTableName}
STORED AS ORC
AS
SELECT * FROM ${hiveconf:customerDemographicsTableName}_textFormat
;


!echo Load text data into ORC table: ${hiveconf:dateTableName};
CREATE TABLE IF NOT EXISTS ${hiveconf:dateTableName}
STORED AS ORC
AS
SELECT * FROM ${hiveconf:dateTableName}_textFormat
;


!echo Load text data into ORC table: ${hiveconf:warehouseTableName};
CREATE TABLE IF NOT EXISTS ${hiveconf:warehouseTableName}
STORED AS ORC
AS
SELECT * FROM ${hiveconf:warehouseTableName}_textFormat
;


!echo Load text data into ORC table: ${hiveconf:shipModeTableName};
CREATE TABLE IF NOT EXISTS ${hiveconf:shipModeTableName}
STORED AS ORC
AS
SELECT * FROM ${hiveconf:shipModeTableName}_textFormat
;


!echo Load text data into ORC table: ${hiveconf:timeTableName};
CREATE TABLE IF NOT EXISTS ${hiveconf:timeTableName}
STORED AS ORC
AS
SELECT * FROM ${hiveconf:timeTableName}_textFormat
;


!echo Load text data into ORC table: ${hiveconf:reasonTableName};
CREATE TABLE IF NOT EXISTS ${hiveconf:reasonTableName}
STORED AS ORC
AS
SELECT * FROM ${hiveconf:reasonTableName}_textFormat
;


!echo Load text data into ORC table: ${hiveconf:incomeTableName};
CREATE TABLE IF NOT EXISTS ${hiveconf:incomeTableName}
STORED AS ORC
AS
SELECT * FROM ${hiveconf:incomeTableName}_textFormat
;


!echo Load text data into ORC table: ${hiveconf:storeTableName};
CREATE TABLE IF NOT EXISTS ${hiveconf:storeTableName}
STORED AS ORC
AS
SELECT * FROM ${hiveconf:storeTableName}_textFormat
;


!echo Load text data into ORC table: ${hiveconf:webSiteTableName};
CREATE TABLE IF NOT EXISTS ${hiveconf:webSiteTableName}
STORED AS ORC
AS
SELECT * FROM ${hiveconf:webSiteTableName}_textFormat
;


!echo Load text data into ORC table: ${hiveconf:householdDemographicsTableName};
CREATE TABLE IF NOT EXISTS ${hiveconf:householdDemographicsTableName}
STORED AS ORC
AS
SELECT * FROM ${hiveconf:householdDemographicsTableName}_textFormat
;


!echo Load text data into ORC table: ${hiveconf:webPageTableName};
CREATE TABLE IF NOT EXISTS ${hiveconf:webPageTableName}
STORED AS ORC
AS
SELECT * FROM ${hiveconf:webPageTableName}_textFormat
;


!echo Load text data into ORC table: ${hiveconf:promotionTableName};
CREATE TABLE IF NOT EXISTS ${hiveconf:promotionTableName}
STORED AS ORC
AS
SELECT * FROM ${hiveconf:promotionTableName}_textFormat
;


!echo Load text data into ORC table: ${hiveconf:customerTableName};
CREATE TABLE IF NOT EXISTS ${hiveconf:customerTableName}
STORED AS ORC
AS
SELECT * FROM ${hiveconf:customerTableName}_textFormat
;


!echo Load text data into ORC table: ${hiveconf:customerAddressTableName};
CREATE TABLE IF NOT EXISTS ${hiveconf:customerAddressTableName}
STORED AS ORC
AS
SELECT * FROM ${hiveconf:customerAddressTableName}_textFormat
;


!echo Load text data into ORC table: ${hiveconf:inventoryTableName};
CREATE TABLE IF NOT EXISTS ${hiveconf:inventoryTableName}
STORED AS ORC
AS
SELECT * FROM ${hiveconf:inventoryTableName}_textFormat
;


!echo Load text data into ORC table: ${hiveconf:itemTableName};
CREATE TABLE IF NOT EXISTS ${hiveconf:itemTableName}
STORED AS ORC
AS
SELECT * FROM ${hiveconf:itemTableName}_textFormat
;


!echo Load text data into ORC table: ${hiveconf:storeSalesTableName};
CREATE TABLE IF NOT EXISTS ${hiveconf:storeSalesTableName}
STORED AS ORC
AS
SELECT * FROM ${hiveconf:storeSalesTableName}_textFormat
;


!echo Load text data into ORC table: ${hiveconf:storeReturnsTableName};
CREATE TABLE IF NOT EXISTS ${hiveconf:storeReturnsTableName}
STORED AS ORC
AS
SELECT * FROM ${hiveconf:storeReturnsTableName}_textFormat
;


!echo Load text data into ORC table: ${hiveconf:webSalesTableName};
CREATE TABLE IF NOT EXISTS ${hiveconf:webSalesTableName}
STORED AS ORC
AS
SELECT * FROM ${hiveconf:webSalesTableName}_textFormat
;


!echo Load text data into ORC table: ${hiveconf:webReturnsTableName};
CREATE TABLE IF NOT EXISTS ${hiveconf:webReturnsTableName}
STORED AS ORC
AS
SELECT * FROM ${hiveconf:webReturnsTableName}_textFormat
;


!echo Load text data into ORC table: ${hiveconf:marketPricesTableName};
CREATE TABLE IF NOT EXISTS ${hiveconf:marketPricesTableName}
STORED AS ORC
AS
SELECT * FROM ${hiveconf:marketPricesTableName}_textFormat
;


!echo Load text data into ORC table: ${hiveconf:clickstreamsTableName};
CREATE TABLE IF NOT EXISTS ${hiveconf:clickstreamsTableName}
STORED AS ORC
AS
SELECT * FROM ${hiveconf:clickstreamsTableName}_textFormat
;


!echo Load text data into ORC table: ${hiveconf:reviewsTableName};
CREATE TABLE IF NOT EXISTS ${hiveconf:reviewsTableName}
STORED AS ORC
AS
SELECT * FROM ${hiveconf:reviewsTableName}_textFormat
;


--- Cleanup 
!echo Dropping all temporary *_textFormat tables;

DROP TABLE IF EXISTS ${hiveconf:customerTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:customerAddressTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:customerDemographicsTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:dateTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:householdDemographicsTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:incomeTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:itemTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:promotionTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:reasonTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:shipModeTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:storeTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:timeTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:warehouseTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:webSiteTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:webPageTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:inventoryTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:storeSalesTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:storeReturnsTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:webSalesTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:webReturnsTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:marketPricesTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:clickstreamsTableName}_textFormat;
DROP TABLE IF EXISTS ${hiveconf:reviewsTableName}_textFormat;

!echo Dropping all temporary *_textFormat tables: DONE;
