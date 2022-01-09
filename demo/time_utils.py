import time

from demo.config import DEMO_CONFIG


def force_min_response_time(f, min_response_time=DEMO_CONFIG.MIN_RESPONSE_TIME):
	def inner(*args, **kwargs):
		start_time = time.time()
		result = f(*args, **kwargs)

		end_time = time.time()
		if end_time - start_time < min_response_time:
			time.sleep(min_response_time - (end_time - start_time))

		return result

	return inner
