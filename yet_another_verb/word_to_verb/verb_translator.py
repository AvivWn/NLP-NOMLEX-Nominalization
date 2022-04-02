import abc


class VerbTranslator(abc.ABC):
	@abc.abstractmethod
	def is_transable(self, word: str) -> bool:
		pass

	@abc.abstractmethod
	def translate(self, word: str) -> str:
		pass
