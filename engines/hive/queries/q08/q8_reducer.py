#"INTEL CONFIDENTIAL"
#Copyright 2015  Intel Corporation All Rights Reserved. 
#
#The source code contained or described herein and all documents related to the source code ("Material") are owned by Intel Corporation or its suppliers or licensors. Title to the Material remains with Intel Corporation or its suppliers and licensors. The Material contains trade secrets and proprietary and confidential information of Intel or its suppliers and licensors. The Material is protected by worldwide copyright and trade secret laws and treaty provisions. No part of the Material may be used, copied, reproduced, modified, published, uploaded, posted, transmitted, distributed, or disclosed in any way without Intel's prior express written permission.
#
#No license under any patent, copyright, trade secret or other intellectual property right is granted to or conferred upon you by disclosure or delivery of the Materials, either expressly, by implication, inducement, estoppel or otherwise. Any license under such intellectual property rights must be express and approved by Intel in writing.

import sys
import logging
import traceback
import os
import time
from time import strftime

category=sys.argv[1] 

'''
To test this script exec:
#intput format tab separated with \t: uid\tc_date\tc_time\tsales_sk\twpt
echo -e "1\t1234\t1234\t1234\treview
1\t1235\t1235\t1234\treview
2\t1234\t1234\t234\treview
2\t1235\t1235\t234\treview" | python q8_reducer.py "review"
'''

def npath(vals):
	#vals ((int(c_date), int(c_time),sales_sk, wpt)
	vals.sort()
	ready = 0
	for val in vals:
		if ready == 0 and val[3] == category:
			ready = 1
			continue
		
		if ready == 1 and val[2] != '\N':
			c_date = val[0]
			sales_sk= val[2]
			print "%s\t%s" % (c_date, sales_sk)
			ready = 0

if __name__ == "__main__":
	line = ''
	try:
		current_key = ''
		vals = []
		#partition by uid
		#order by c_date, c_time
		#The plan: create an vals[] per uid, with layout (c_date, c_time, sales_sk, wpt)
		
		for line in sys.stdin:
			#print("line:" + line + "\n")
			uid, c_date, c_time, sales_sk, wpt = line.strip().split("\t")

			#ignore date time parsing errors
			try:
				c_date = int(c_date)
				c_time = int(c_time)
			except ValueError:
				c_date = -1
				c_time = -1
				continue

			if current_key == '' :
				current_key = uid
				vals.append((c_date, c_time, sales_sk, wpt))

			elif current_key == uid :
				vals.append((c_date, c_time, sales_sk, wpt))

			elif current_key != uid :
				npath(vals)
				vals = []
				current_key = uid
				vals.append((c_date, c_time, sales_sk, wpt))

		npath(vals)

	except:
	 ## should only happen if input format is not correct, like 4 instead of 5 tab separated values
		logging.basicConfig(level=logging.DEBUG, filename=strftime("/tmp/bigbench_q8_reducer_%Y%m%d-%H%M%S.log"))
		logging.info('category: ' +category )
		logging.info("line from hive: \"" + line + "\"")
		logging.exception("Oops:")
		sys.exit(1)
