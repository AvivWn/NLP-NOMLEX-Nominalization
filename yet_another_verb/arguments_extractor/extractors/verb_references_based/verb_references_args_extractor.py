from typing import Optional

import torch

from yet_another_verb.arguments_extractor.extraction import Extractions, Extraction, ArgumentType
from yet_another_verb.arguments_extractor.extractors.dep_related_args_extractor import DepRelatedArgsExtractor
from yet_another_verb.arguments_extractor.extractors.verb_references_based.candidate_fetchers import \
	CandidatesFetcher, DependencyCandidatesFetcher
from yet_another_verb.arguments_extractor.extractors.verb_references_based.argument_labelers import \
	ArgumentsLabeler, ArgumentsLabelerByArgumentKNN, ArgumentsLabelerByAVG, ArgumentsLabelerByExtractionKNN
from yet_another_verb.arguments_extractor.extractors.verb_references_based.method_params import MethodParams
from yet_another_verb.arguments_extractor.extractors.verb_references_based.verb_references.references_corpus import \
	ReferencesByPredicate
from yet_another_verb.dependency_parsing import POSTag
from yet_another_verb.dependency_parsing.dependency_parser.dependency_parser import DependencyParser
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText
from yet_another_verb.dependency_parsing.dependency_parser.parsed_word import ParsedWord
from yet_another_verb.sentence_encoding.argument_encoding.arg_encoder import ArgumentEncoder
from yet_another_verb.word_to_verb.verb_translator import VerbTranslator


class VerbReferencesArgsExtractor(DepRelatedArgsExtractor):
	def __init__(
			self,
			dependency_parser: DependencyParser,
			verb_translator: VerbTranslator,
			arg_encoder: ArgumentEncoder,
			candidates_fetcher: CandidatesFetcher,
			arguments_labeler: ArgumentsLabeler,
			method_params: MethodParams,
			references_by_verb: ReferencesByPredicate,
			**kwargs
	):
		super().__init__(dependency_parser, **kwargs)
		self.verb_translator = verb_translator
		self.arg_encoder = arg_encoder
		self.candidates_fetcher = candidates_fetcher
		self.arguments_labeler = arguments_labeler
		self.method_params = method_params
		self.references_by_verb = references_by_verb

	def _get_appropriate_verb(self, word: ParsedWord) -> str:
		if word.pos == POSTag.VERB:
			return word.lemma

		return self.verb_translator.translate(word.lemma, word.pos)

	def extract(self, word_idx: int, words: ParsedText) -> Optional[Extractions]:
		word = words[word_idx]
		verb = self._get_appropriate_verb(word)

		if verb not in self.references_by_verb:
			return []

		verb_references = self.references_by_verb[verb].get_filtered_references(
			self.method_params.references_amount, self.method_params.arg_types)
		arg_candidates = self.candidates_fetcher.fetch_arguments(word_idx, words, self.method_params)

		for arg in arg_candidates:
			encoding = self.arg_encoder.encode(words.words, arg)
			arg.encoding = encoding.numpy() if isinstance(encoding, torch.Tensor) else encoding

		if len(arg_candidates) == 0:
			return []

		self.arguments_labeler.label_arguments(arg_candidates, words, verb_references, self.method_params)
		args = [arg for arg in arg_candidates if arg.arg_type != ArgumentType.REDUNDANT]

		if len(args) == 0:
			return []

		# Construct extraction from the labels
		return [
			Extraction(
				words=words, predicate_idx=word_idx,
				predicate_lemma=word.lemma,
				predicate_postag=word.pos,
				args=args)]


class DependencyArgAVGArgsExtractor(VerbReferencesArgsExtractor):
	def __init__(
			self,
			dependency_parser: DependencyParser,
			verb_translator: VerbTranslator,
			arg_encoder: ArgumentEncoder,
			method_params: MethodParams,
			references_by_verb: ReferencesByPredicate,
			**kwargs
	):
		super().__init__(
			dependency_parser, verb_translator, arg_encoder, DependencyCandidatesFetcher(),
			ArgumentsLabelerByAVG(), method_params, references_by_verb, **kwargs)


class DependencyArgKNNArgsExtractor(VerbReferencesArgsExtractor):
	def __init__(
			self,
			dependency_parser: DependencyParser,
			verb_translator: VerbTranslator,
			arg_encoder: ArgumentEncoder,
			method_params: MethodParams,
			references_by_verb: ReferencesByPredicate,
			**kwargs
	):
		super().__init__(
			dependency_parser, verb_translator, arg_encoder, DependencyCandidatesFetcher(),
			ArgumentsLabelerByArgumentKNN(), method_params, references_by_verb, **kwargs)


class DependencyExtKNNArgsExtractor(VerbReferencesArgsExtractor):
	def __init__(
			self,
			dependency_parser: DependencyParser,
			verb_translator: VerbTranslator,
			arg_encoder: ArgumentEncoder,
			method_params: MethodParams,
			references_by_verb: ReferencesByPredicate,
			**kwargs
	):
		super().__init__(
			dependency_parser, verb_translator, arg_encoder, DependencyCandidatesFetcher(),
			ArgumentsLabelerByExtractionKNN(), method_params, references_by_verb, **kwargs)
