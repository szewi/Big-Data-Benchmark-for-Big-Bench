

set customerTableName=customer;
set customerAddressTableName=customer_address;
set customerDemographicsTableName=customer_demographics;
set dateTableName=date_dim;
set dbgenTableName=dbgen_version;
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


DROP TABLE IF EXISTS ${hiveconf:customerTableName};
DROP TABLE IF EXISTS ${hiveconf:customerAddressTableName};
DROP TABLE IF EXISTS ${hiveconf:customerDemographicsTableName};
DROP TABLE IF EXISTS ${hiveconf:dateTableName};
DROP TABLE IF EXISTS ${hiveconf:dbgenTableName};
DROP TABLE IF EXISTS ${hiveconf:householdDemographicsTableName};
DROP TABLE IF EXISTS ${hiveconf:incomeTableName};
DROP TABLE IF EXISTS ${hiveconf:itemTableName};
DROP TABLE IF EXISTS ${hiveconf:promotionTableName};
DROP TABLE IF EXISTS ${hiveconf:reasonTableName};
DROP TABLE IF EXISTS ${hiveconf:shipModeTableName};
DROP TABLE IF EXISTS ${hiveconf:storeTableName};
DROP TABLE IF EXISTS ${hiveconf:timeTableName};
DROP TABLE IF EXISTS ${hiveconf:warehouseTableName};
DROP TABLE IF EXISTS ${hiveconf:webSiteTableName};
DROP TABLE IF EXISTS ${hiveconf:webPageTableName};
DROP TABLE IF EXISTS ${hiveconf:inventoryTableName};
DROP TABLE IF EXISTS ${hiveconf:storeSalesTableName};
DROP TABLE IF EXISTS ${hiveconf:storeReturnsTableName};
DROP TABLE IF EXISTS ${hiveconf:webSalesTableName};
DROP TABLE IF EXISTS ${hiveconf:webReturnsTableName};

DROP TABLE IF EXISTS ${hiveconf:marketPricesTableName};
DROP TABLE IF EXISTS ${hiveconf:clickstreamsTableName};
DROP TABLE IF EXISTS ${hiveconf:reviewsTableName};
