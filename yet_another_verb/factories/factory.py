import abc
from argparse import ArgumentParser
from typing import Optional


class Factory(abc.ABC):
	@abc.abstractmethod
	def __call__(self):
		pass

	@staticmethod
	def _expand_optional_parser(arg_parser: Optional[ArgumentParser] = None) -> ArgumentParser:
		return ArgumentParser(parents=[arg_parser], add_help=False) if arg_parser else ArgumentParser()
