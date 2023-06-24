from collections import Counter
from typing import List, Optional, Set

import os
from tqdm import tqdm
from pony.orm import db_session

from yet_another_verb.arguments_extractor.args_extractor import ArgsExtractor
from yet_another_verb.arguments_extractor.extraction import MultiWordExtraction
from yet_another_verb.arguments_extractor.extraction.utils.combination import combine_extractions
from yet_another_verb.configuration.extractors_config import NAME_BY_EXTRACTOR
from yet_another_verb.data_handling import ParsedBinFileHandler
from yet_another_verb.data_handling.dataset_creator import DatasetCreator
from yet_another_verb.data_handling.db.communicators.sqlite_communicator import SQLiteCommunicator
from yet_another_verb.dependency_parsing.dependency_parser.parsed_bin import ParsedBin
from yet_another_verb.sentence_encoding.argument_encoding.arg_encoder import ArgumentEncoder
from yet_another_verb.sentence_encoding.argument_encoding.encoding_level import EncodingLevel, encoder_by_level
from yet_another_verb.data_handling.db.encoded_extractions.queries import get_extractor, get_encoder, get_sentence, \
	get_predicate_in_sentence, get_predicate, get_extracted_predicates, get_extracted_indices_in_sentence, get_parser, \
	insert_encoded_arguments
from yet_another_verb.data_handling.db.encoded_extractions.structure import encoded_extractions_db, Parser, \
	Parsing, Sentence, Extractor, Encoder as DBEncoder
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText
from yet_another_verb.dependency_parsing import POSTag
from yet_another_verb.dependency_parsing.postagged_word import POSTaggedWord
from yet_another_verb.sentence_encoding.encoder import Encoder
from yet_another_verb.word_to_verb.verb_translator import VerbTranslator
from yet_another_verb.configuration.encoding_config import FRAMEWORK_BY_ENCODER
from yet_another_verb.configuration.parsing_config import ENGINE_BY_PARSER
from yet_another_verb.utils.debug_utils import timeit


class EncodedExtractionsCreator(DatasetCreator):
	def __init__(
			self, in_dataset_path: str,
			args_extractor: ArgsExtractor,
			verb_translator: VerbTranslator,
			encoder: Encoder,
			encoding_level: EncodingLevel = EncodingLevel.HEAD_IDX_IN_SENTENCE_CONTEXT,
			limited_postags: List[POSTag] = None,
			limited_verbs: List[str] = None,
			dataset_size=None, **kwargs
	):
		super().__init__(dataset_size)
		self.in_dataset_path = in_dataset_path

		self.args_extractor = args_extractor
		self.verb_translator = verb_translator

		self.encoder = encoder
		self.encoding_level = encoding_level

		self.limited_postags = limited_postags
		self.limited_verbs = limited_verbs

	def _is_relevant(self, postagged_word: POSTaggedWord, postagged_verb_counter: Optional[Counter] = None) -> bool:
		verb = self.verb_translator.translate(postagged_word.word, postagged_word.postag)

		if self.limited_verbs is not None and verb not in self.limited_verbs:
			return False

		if self.limited_postags is not None and postagged_word.postag not in self.limited_postags:
			return False

		if postagged_verb_counter is not None and POSTaggedWord(verb, postagged_word.postag) not in postagged_verb_counter:
			return False

		return True

	def _get_potential_indices(self, doc: ParsedText, extractor_entity: Extractor, postagged_verb_counter: Counter) -> Set[int]:
		limited_indices = {w.i for w in doc if self._is_relevant(POSTaggedWord(w.lemma, w.pos), postagged_verb_counter)}

		if len(limited_indices) == 0:
			return limited_indices

		sentence_entity = get_sentence(doc.text)
		if sentence_entity is not None:
			already_extracted_indices = get_extracted_indices_in_sentence(sentence_entity, extractor_entity)
			limited_indices = limited_indices.difference(already_extracted_indices)

		return limited_indices

	def _get_postagged_verb_counter(self, extractor_entity: Extractor) -> Counter:
		postaged_verbs = set()

		for predicate in self.verb_translator.supported_words:
			if self._is_relevant(predicate):
				verb = self.verb_translator.translate(predicate.word, predicate.postag)
				postaged_verbs.add(POSTaggedWord(verb, predicate.postag))

		postagged_verb_counter = Counter({w: 0 for w in postaged_verbs})

		# Update with existing extractions
		extracted_predicates = get_extracted_predicates(extractor_entity)
		for predicate_in_sentence in extracted_predicates:
			predicate_entity = predicate_in_sentence.predicate
			postag_entity = predicate_entity.part_of_speech
			postagged_verb = POSTaggedWord(predicate_entity.verb.lemma, postag_entity.part_of_speech)

			if postagged_verb in postagged_verb_counter:
				postagged_verb_counter[postagged_verb] += 1

			if self.has_reached_size(postagged_verb_counter[postagged_verb]):
				postagged_verb_counter.pop(postagged_verb)

		return postagged_verb_counter

	@staticmethod
	def _store_parsing(doc: ParsedText, sentence_entity: Sentence, parser_entity: Parser, dependency_parser):
		if Parsing.get(sentence=sentence_entity, parser=parser_entity) is None:
			parsed_bin = dependency_parser.generate_parsed_bin()
			parsed_bin.add(doc)
			binary = parsed_bin.to_bytes()
			Parsing(sentence=sentence_entity, parser=parser_entity, binary=binary)

	def _store_extractions_by_predicates(
			self, doc: ParsedText, multi_word_extraction: MultiWordExtraction,
			extractor_entity: Extractor, sentence_entity: Sentence,
			encoder_entity: DBEncoder, arg_encoder: ArgumentEncoder,
			postagged_verb_counter: Counter
	):
		for predicate_idx, extractions in multi_word_extraction.extractions_per_idx.items():
			predicate = doc[predicate_idx]
			verb = self.verb_translator.translate(predicate.lemma, predicate.pos)
			postagged_verb = POSTaggedWord(verb, predicate.pos)

			if postagged_verb not in postagged_verb_counter:
				continue

			postagged_verb_counter[postagged_verb] += 1

			predicate_entity = get_predicate(verb, predicate.pos, predicate.lemma, generate_missing=True)
			predicate_in_sentence = get_predicate_in_sentence(sentence_entity, predicate_entity, predicate.i, generate_missing=True)

			combined_extraction = combine_extractions(extractions, safe_combine=False)
			insert_encoded_arguments(combined_extraction.args, extractor_entity, predicate_in_sentence, encoder_entity, arg_encoder)

			if self.has_reached_size(postagged_verb_counter[postagged_verb]):
				postagged_verb_counter.pop(postagged_verb)

	@db_session
	def _store_encoded_extractions(self, db_communicator: SQLiteCommunicator, parsed_bin: ParsedBin):
		dependency_parser = parsed_bin.parser
		parser_entity = get_parser(
			ENGINE_BY_PARSER[type(dependency_parser)], dependency_parser.name, generate_missing=True)
		extractor_entity = get_extractor(NAME_BY_EXTRACTOR[type(self.args_extractor)], parser_entity, generate_missing=True)
		encoder_entity = get_encoder(
			FRAMEWORK_BY_ENCODER[type(self.encoder)], self.encoder.name, self.encoding_level,
			parser_entity, generate_missing=True)

		postagged_verb_counter = self._get_postagged_verb_counter(extractor_entity)

		loop_status = tqdm(parsed_bin.get_parsed_texts(), leave=False)
		for doc in loop_status:
			limited_indices = self._get_potential_indices(doc, extractor_entity, postagged_verb_counter)
			multi_word_extraction = timeit(self.args_extractor.extract_multiword)(doc, limited_indices=limited_indices)
			if len(multi_word_extraction.extractions_per_idx) == 0:
				continue

			sentence_entity = timeit(get_sentence)(doc.text, generate_missing=True)
			arg_encoder = encoder_by_level.get(self.encoding_level)(words=doc, encoder=self.encoder)

			timeit(self._store_parsing)(doc, sentence_entity, parser_entity, dependency_parser)
			timeit(self._store_extractions_by_predicates)(
				doc, multi_word_extraction, extractor_entity, sentence_entity,
				encoder_entity, arg_encoder, postagged_verb_counter)

			timeit(db_communicator.commit)()
			if len(postagged_verb_counter) == 0:
				break

			loop_status.set_description(f"Missing: {len(postagged_verb_counter)}")

	def append_dataset(self, out_dataset_path):
		in_parsed_bin = ParsedBinFileHandler().load(self.in_dataset_path)

		create_db = not self.is_dataset_exist(out_dataset_path)
		db_communicator = SQLiteCommunicator(encoded_extractions_db, out_dataset_path, create_db=create_db)
		db_communicator.generate_mapping()

		self._store_encoded_extractions(db_communicator, in_parsed_bin)
		db_communicator.disconnect()

	def create_dataset(self, out_dataset_path):
		if self.is_dataset_exist(out_dataset_path):
			os.remove(out_dataset_path)

		self.append_dataset(out_dataset_path)
