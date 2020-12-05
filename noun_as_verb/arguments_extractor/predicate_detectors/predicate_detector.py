import abc


class PredicateDetector(abc.ABC):
	def __init__(self):
		pass

	def is_predicate(self, token):
		...
