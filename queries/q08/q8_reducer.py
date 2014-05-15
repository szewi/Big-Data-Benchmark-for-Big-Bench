import sys
import logging
import traceback
import os


def npath(vals):
	#vals ((int(c_date), int(c_time),sales_sk, wpt)
	vals.sort()
	ready = 0
	for val in vals:
		if ready == 0 and val[3] == 'review':
			ready = 1
			continue
		
		if ready == 1 and val[2] != '\N':
			c_date = val[0]
			sales_sk= val[2]
			#if c_date > 36403 and c_date <36403+365:
			#	print "%s\t%s" % (c_date, sales_sk)
			print "%s\t%s" % (c_date, sales_sk)
			ready = 0

if __name__ == "__main__":
	logging.basicConfig(level=logging.DEBUG, filename='/tmp/q8reducerErr.log')

	try:
		current_key = ''
		vals = []
		#partition by uid
		#order by c_date, c_time
		#The plan: create an vals[] per uid, with layout (c_date, c_time, sales_sk, wpt)

		for line in sys.stdin:
			#print("line:" + line + "\n")
			uid, c_date, c_time, sales_sk, wpt = line.strip().split("\t")

			try:
				c_date = int(c_date)
				c_time = int(c_time)
			except ValueError:
				c_date = -1
				c_time = -1
				continue

			if current_key == '' :
				current_key = uid
				vals.append((int(c_date), int(c_time),sales_sk, wpt))

			elif current_key == uid :
				vals.append((int(c_date), int(c_time),sales_sk, wpt))

			elif current_key != uid :
				npath(vals)
				vals = []
				current_key = uid
				vals.append((int(c_date), int(c_time),sales_sk, wpt))

		npath(vals)	except:
	    logging.exception("Oops:")
