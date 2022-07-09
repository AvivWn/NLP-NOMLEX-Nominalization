import abc
from typing import Set


class VerbTranslator(abc.ABC):
	@abc.abstractmethod
	def is_transable(self, word: str) -> bool:
		pass

	@abc.abstractmethod
	def translate(self, word: str) -> str:
		pass

	@property
	@abc.abstractmethod
	def words(self) -> Set[str]:
		pass

	@property
	@abc.abstractmethod
	def verbs(self) -> Set[str]:
		pass
