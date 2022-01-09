import typeguard
import spacy
from spacy.util import compile_infix_regex
from spacy.tokenizer import Tokenizer
from spacy.tokens import Token, Doc

from yet_another_verb.dependency_parsing.spacy.spacy_parsed_text import SpacyParsedText
from yet_another_verb.dependency_parsing.dependency_parser.input_text import InputText
from yet_another_verb.dependency_parsing.dependency_parser.dependency_parser import DependencyParser

spacy.util.fix_random_seed()
Token.set_extension("subtree_text", getter=lambda token: " ".join([node.text for node in token.subtree]))
Token.set_extension("subtree_indices", getter=lambda token: [node.i for node in token.subtree])

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class UDParser(DependencyParser):
	def __init__(self, parser_name: str):
		self._parser = spacy.load(parser_name)
		self._parser.tokenizer = self._create_custom_tokenizer()

	def __call__(self, text: InputText, disable=None):
		return self.parse(text, disable)

	def _create_custom_tokenizer(self):
		infixes = self._parser.Defaults.infixes
		infixes = [x for x in infixes if '-|–|—|--|---|——|~' not in x]  # Remove "-" between letters rule
		infix_re = compile_infix_regex(infixes)

		return Tokenizer(
			self._parser.vocab,
			prefix_search=self._parser.tokenizer.prefix_search,
			suffix_search=self._parser.tokenizer.suffix_search,
			infix_finditer=infix_re.finditer,
			token_match=self._parser.tokenizer.token_match,
			rules=self._parser.Defaults.tokenizer_exceptions
		)

	@typeguard.typechecked
	def parse(self, text: InputText, disable=None) -> SpacyParsedText:
		if isinstance(text, SpacyParsedText):
			return text

		if isinstance(text, Doc):
			return SpacyParsedText(text)

		if disable is None:
			disable = []

		if isinstance(text, str):
			d = SpacyParsedText(self._parser(text, disable=disable))
			print([x.dep for x in d])
			print([x.head for x in d])
			print([x.tag for x in d])
			return d

		# parse a tokenized sentence
		doc = self._parser.tokenizer.tokens_from_list(text)
		for name, proc in self._parser.pipeline:
			if name not in disable:
				doc = proc(doc)

		return SpacyParsedText(doc)

	@property
	def vocab(self):
		return self._parser.vocab
