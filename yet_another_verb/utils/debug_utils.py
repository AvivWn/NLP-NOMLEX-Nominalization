import time


def timeit(method):
	def timed(*args, **kw):
		ts = time.time()
		result = method(*args, **kw)
		te = time.time()
		if 'log_time' in kw:
			name = kw.get('log_name', method.__name__.upper())
			kw['log_time'][name] = int((te - ts) * 1000)
		else:
			print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
			return result

	return timed


def print_extraction(extraction_repr):
	for predicate, extractions in extraction_repr.items():
		if not extractions:
			continue

		print(predicate + ":")

		for e in extractions:
			print(" " * 4 + str(e))
