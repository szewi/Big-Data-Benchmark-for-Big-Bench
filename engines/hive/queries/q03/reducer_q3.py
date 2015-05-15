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

def npath(vals):
	vals.sort()
	last_viewed_item = -1
	last_viewed_date = -1
	for i in vals:
		if i[2] == '\N' and i[1] != '\N':
			last_viewed_item = i[1]
			last_viewed_date = i[0]
		elif i[2] != '\N' and i[1] != '\N' and last_viewed_item > -1 and last_viewed_date >= (i[0]- days_param ) :
			print "%s\t%s" % (last_viewed_item, i[1])
			last_viewed_item = i[1]
			last_viewed_date = i[0]
		

if __name__ == "__main__":

	last_user = ''
	vals = []

	for line in sys.stdin:
		user, wcs_date, item_key, sale = line.strip().split("\t")

		try:
			wcs_date = long(wcs_date)
			item_key = long(item_key)
		except ValueError:
			wcs_date = -1
			item_key = -1
			continue


		if last_user == user :
			vals.append((wcs_date, item_key, sale))
		else :
			npath(vals)
			vals = []
			last_user = user;
			vals.append((wcs_date, item_key, sale))
	npath(vals)	
	
