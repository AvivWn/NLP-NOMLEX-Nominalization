import abc
from typing import List, Iterator


class ParsedWord(abc.ABC):
	@abc.abstractmethod
	def __len__(self) -> int: pass

	@abc.abstractmethod
	def __str__(self) -> str: pass

	@abc.abstractmethod
	def __repr__(self) -> str: pass

	@abc.abstractmethod
	def __unicode__(self) -> str: pass

	@abc.abstractmethod
	def __bytes__(self) -> bytes: pass

	@abc.abstractmethod
	def __hash__(self) -> int: pass

	@abc.abstractmethod
	def __eq__(self, other) -> bool: pass

	@property
	@abc.abstractmethod
	def i(self) -> int: pass

	@property
	@abc.abstractmethod
	def subtree(self) -> Iterator['ParsedWord']: pass

	@property
	@abc.abstractmethod
	def children(self) -> Iterator['ParsedWord']: pass

	@property
	@abc.abstractmethod
	def head(self) -> 'ParsedWord': pass

	@property
	@abc.abstractmethod
	def dep(self) -> str: pass

	@property
	@abc.abstractmethod
	def text(self) -> str: pass

	@property
	@abc.abstractmethod
	def lemma(self) -> str: pass

	@property
	@abc.abstractmethod
	def tag(self) -> str: pass

	@property
	@abc.abstractmethod
	def pos(self) -> str: pass

	@property
	def subtree_text(self) -> str:
		return " ".join([node.text for node in self.subtree])

	@property
	def subtree_indices(self) -> List[int]:
		return [node.i for node in self.subtree]
