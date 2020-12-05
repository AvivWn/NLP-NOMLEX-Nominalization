import abc


class ExtractionRepresentation(abc.ABC):
	def __init__(self):
		pass

	@abc.abstractmethod
	def represent(self, extraction: Exception, trim_arguments=True):
		...

	def represent_dict(self, predicate2extractions, trim_arguments=True):
		dict_repr = {}

		for predicate, extractions in predicate2extractions.items():
			extractions_repr = [self.represent(e, trim_arguments) for e in extractions]
			dict_repr[predicate.get_token()] = extractions_repr

		return dict_repr
