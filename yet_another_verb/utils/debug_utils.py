import time

from yet_another_verb.utils.print_utils import print_if_verbose


def timeit(method):
	def timed(*args, **kw):
		ts = time.time()
		result = method(*args, **kw)
		te = time.time()
		if 'log_time' in kw:
			name = kw.get('log_name', method.__name__.upper())
			kw['log_time'][name] = int((te - ts) * 1000)
		else:
			print_if_verbose('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
			return result

	return timed
