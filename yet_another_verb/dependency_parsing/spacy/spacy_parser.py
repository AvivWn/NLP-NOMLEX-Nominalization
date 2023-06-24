import os

import typeguard
import spacy
from spacy.util import compile_infix_regex
from spacy.tokenizer import Tokenizer
from spacy.tokens import Token, Doc

from yet_another_verb.dependency_parsing.dependency_parser.parsed_bin import ParsedBin
from yet_another_verb.dependency_parsing.spacy.spacy_parsed_text import SpacyParsedText
from yet_another_verb.dependency_parsing.dependency_parser.input_text import InputText
from yet_another_verb.dependency_parsing.dependency_parser.dependency_parser import DependencyParser
from yet_another_verb.utils.debug_utils import timeit
from yet_another_verb.utils.print_utils import print_if_verbose

spacy.util.fix_random_seed()
Token.set_extension("subtree_text", getter=lambda token: " ".join([node.text for node in token.subtree]))
Token.set_extension("subtree_indices", getter=lambda token: [node.i for node in token.subtree])

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class SpacyParser(DependencyParser):
	def __init__(self, parser_name: str, **kwargs):
		self.parser_name = parser_name
		self._parser: spacy.Language = timeit(spacy.load)(self.parser_name)
		self._parser.tokenizer = self._create_custom_tokenizer()

	def __call__(self, text: InputText, disable=None):
		return self.parse(text, disable)

	@property
	def name(self) -> str:
		return self.parser_name

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

			print_if_verbose("Dependencies:", [x.dep for x in d])
			print_if_verbose("Heads:", [x.head for x in d])
			print_if_verbose("Postags:", [x.tag for x in d])

			return d

		# parse a tokenized sentence
		assert isinstance(text, list)
		doc = Doc(self.vocab, words=text)
		for name, proc in self._parser.pipeline:
			if name not in disable:
				doc = proc(doc)

		return SpacyParsedText(doc)

	def from_bytes(self, bytes_data: bytes) -> SpacyParsedText:
		return SpacyParsedText(Doc(self.vocab).from_bytes(bytes_data))

	@property
	def vocab(self):
		return self._parser.vocab

	def generate_parsed_bin(self) -> ParsedBin:
		from yet_another_verb.dependency_parsing.spacy.spacy_parsed_bin import SpacyParsedBin
		return SpacyParsedBin(self)
