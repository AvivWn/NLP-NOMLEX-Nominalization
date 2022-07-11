import abc
from typing import Set, Union

from yet_another_verb.dependency_parsing import POSTag, POSTaggedWord


class VerbTranslator(abc.ABC):
	@abc.abstractmethod
	def is_transable(self, word: str, postag: Union[POSTag, str]) -> bool:
		pass

	@abc.abstractmethod
	def translate(self, word: str, postag: Union[POSTag, str]) -> str:
		pass

	@property
	@abc.abstractmethod
	def supported_words(self) -> Set[POSTaggedWord]:
		pass

	@property
	@abc.abstractmethod
	def verbs(self) -> Set[str]:
		pass
