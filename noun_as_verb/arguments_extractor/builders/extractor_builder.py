import abc


class ExtractorBuilder(abc.ABC):
	def __init__(self):
		pass

	@abc.abstractmethod
	def get_predicate_detector(self):
		...

	@abc.abstractmethod
	def get_candidates_finder(self):
		...

	@abc.abstractmethod
	def get_structural_matcher(self):
		...

	@abc.abstractmethod
	def get_extractions_filter(self):
		...
