import abc

from ..extraction.predicate import Predicate


class StructuralMatcher(abc.ABC):
	def __init__(self):
		pass

	@abc.abstractmethod
	def find_structures(self, candidates, predicate: Predicate):
		...
