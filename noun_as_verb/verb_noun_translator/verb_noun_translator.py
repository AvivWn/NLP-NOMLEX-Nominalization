import abc


class VerbNounTranslator(abc.ABC):
	def __init__(self):
		pass

	@abc.abstractmethod
	def get_suitable_verb(self, token):
		...

	def is_contain(self, token):
		...

	def is_predicate(self, token):
		...