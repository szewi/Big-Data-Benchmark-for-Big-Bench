#"INTEL CONFIDENTIAL"
#Copyright 2015  Intel Corporation All Rights Reserved. 
#
#The source code contained or described herein and all documents related to the source code ("Material") are owned by Intel Corporation or its suppliers or licensors. Title to the Material remains with Intel Corporation or its suppliers and licensors. The Material contains trade secrets and proprietary and confidential information of Intel or its suppliers and licensors. The Material is protected by worldwide copyright and trade secret laws and treaty provisions. No part of the Material may be used, copied, reproduced, modified, published, uploaded, posted, transmitted, distributed, or disclosed in any way without Intel's prior express written permission.
#
#No license under any patent, copyright, trade secret or other intellectual property right is granted to or conferred upon you by disclosure or delivery of the Materials, either expressly, by implication, inducement, estoppel or otherwise. Any license under such intellectual property rights must be express and approved by Intel in writing.

import sys
import traceback
import os

days_param = long(sys.argv[1])
last_n_views = long(sys.argv[2])

#Explanation:
#Reducer script logic: iterate through clicks of a user in descending order (most recent click first).if a purchase is found (wcs_sales_sk!=null) display the next 5 clicks if they are within the provided date range (max 10 days before)
#Reducer script selects only:
# * products viewed within 'q03_days_before_purchase' days before the purchase date
# * only the last 5 products that where purchased before a sale	
#
# Limitations of this implementation:
# a newly purchased item resets the "clicks before purchase"
#
# Future:
# This could be circumvented by iterating in ascending click_date order instead of descending and keeping a cyclic buffer of length 'last_n_views' storing the last n clicked_item_sk's and click_dates. Upon finding the next purchase, dump the buffer contents for each buffer item matching the date range. 

if __name__ == "__main__":

	last_user = ''
	vals = []
	last_sale_item =-1
	last_sale_date =-1
	last_viewed_item = -1
	last_viewed_date = -1
	views_since_sale = 0
	
	for line in sys.stdin:
		#values are clustered by user and pre-sorted by wcs_date (done in hive)
		user, wcs_date, item_key, sale_sk = line.strip().split("\t")

		try:
			wcs_date = long(wcs_date)
			item_key = long(item_key)
		except ValueError:
			wcs_date = -1
			item_key = -1
			continue
			
		#new user , reset everything
		if last_user != user :
			last_user = user;
			last_sale_item =-1
			last_sale_date =-1
			last_viewed_item = -1
			last_viewed_date = -1
			views_since_sale =0
			
		#next sale. '\N' is hives encoding of NULL
		if sale_sk != '\N'  :
			last_sale_date = wcs_date
			last_sale_item = item_key
			views_since_sale = 0
		
		elif last_sale_item != -1 and views_since_sale < last_n_views and wcs_date <= last_sale_date  and wcs_date >= ( last_sale_date - days_param ) :
			views_since_sale = views_since_sale + 1
			print "%s\t%s" % (last_sale_item, item_key)

		#else : ignore view clicks out of date range (days_param) after sale and  clicks that exceed the maximum of n clicks after sale (last_n_views)
				
