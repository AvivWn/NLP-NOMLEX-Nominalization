from collections import Counter
from os.path import isdir, join
from os import listdir
from typing import Iterator, List, Optional, Set
from itertools import chain

import os
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from pony.orm import db_session

from yet_another_verb.arguments_extractor.args_extractor import ArgsExtractor
from yet_another_verb.arguments_extractor.extraction import MultiWordExtraction
from yet_another_verb.arguments_extractor.extraction.utils.combination import combine_extractions
from yet_another_verb.data_handling import ParsedBinFileHandler
from yet_another_verb.data_handling.dataset_creator import DatasetCreator
from yet_another_verb.data_handling.db.communicators.sqlite_communicator import SQLiteCommunicator
from yet_another_verb.data_handling.db.encoded_extractions.encodings import EncodingLevel, ENCODER_BY_LEVEL, \
	ArgumentEncoder
from yet_another_verb.data_handling.db.encoded_extractions.queries import get_extractor, get_encoder, get_sentence, \
	get_predicate_in_sentence, get_predicate, get_extracted_predicates, get_extracted_idxs_in_sentence, get_parser, \
	insert_encoded_arguments
from yet_another_verb.data_handling.db.encoded_extractions.structure import encoded_extractions_db, Parser, \
	Parsing, Sentence, Encoder, Extractor
from yet_another_verb.dependency_parsing.dependency_parser.dependency_parser import DependencyParser
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText
from yet_another_verb.dependency_parsing import POSTag, engine_by_parser
from yet_another_verb.dependency_parsing.postagged_word import POSTaggedWord
from yet_another_verb.word_to_verb.verb_translator import VerbTranslator
from yet_another_verb.configuration import EXTRACTORS_CONFIG
from yet_another_verb.utils.debug_utils import timeit


class EncodedExtractionsCreator(DatasetCreator):
	def __init__(
			self, in_dataset_path: str,
			dependency_parser: DependencyParser,
			args_extractor: ArgsExtractor,
			verb_translator: VerbTranslator,
			model_name: str, device: str,
			encoding_level: EncodingLevel = EncodingLevel.HEAD_IDX,
			limited_postags: List[POSTag] = None,
			limited_words: List[str] = None,
			dataset_size=None, **kwargs
	):
		super().__init__(dataset_size)
		self.in_dataset_path = in_dataset_path

		self.dependency_parser = dependency_parser
		self.args_extractor = args_extractor
		self.verb_translator = verb_translator

		self.model_name = model_name
		self.encoding_level = encoding_level
		self.device = device
		self.tokenizer = AutoTokenizer.from_pretrained(model_name)
		self.model = AutoModel.from_pretrained(model_name).to(self.device)
		self.model.eval()

		self.limited_postags = limited_postags
		self.limited_words = limited_words

	def _load_parsed_dataset(self, in_dataset_path):
		in_dataset_paths = [in_dataset_path]
		extend_with_parser_id = True

		if isdir(in_dataset_path):
			in_dataset_paths = [join(in_dataset_path, file_name) for file_name in listdir(in_dataset_path)[:1]]
			extend_with_parser_id = False

		parsed_datasets = []
		for path in in_dataset_paths:
			parsed_bin = ParsedBinFileHandler(self.dependency_parser, extend_with_parser_id=extend_with_parser_id).load(path)
			parsed_datasets.append(parsed_bin.get_parsed_texts())

		return chain(*parsed_datasets)

	def _is_relevant_predicate(self, postagged_word: POSTaggedWord, predicate_counter: Optional[Counter] = None) -> bool:
		if self.limited_words is not None and postagged_word.word not in self.limited_words:
			return False

		if self.limited_postags is not None and postagged_word.postag not in self.limited_postags:
			return False

		if predicate_counter is not None and postagged_word not in predicate_counter:
			return False

		return True

	def _get_potential_idxs(self, doc: ParsedText, extractor_entity: Extractor, predicate_counter: Counter) -> Set[int]:
		limited_idxs = {w.i for w in doc if self._is_relevant_predicate(POSTaggedWord(w.lemma, w.pos), predicate_counter)}

		if len(limited_idxs) == 0:
			return limited_idxs

		sentence_entity = get_sentence(doc.text)
		if sentence_entity is not None:
			already_extracted_idxs = get_extracted_idxs_in_sentence(sentence_entity, extractor_entity)
			limited_idxs = limited_idxs.difference(already_extracted_idxs)

		return limited_idxs

	def _get_predicate_counter(self, extractor_entity: Extractor) -> Counter:
		predicate_counter = Counter({w: 0 for w in self.verb_translator.supported_words if self._is_relevant_predicate(w)})

		# Update with existing extractions
		extracted_predicates = get_extracted_predicates(extractor_entity)
		for predicate_in_sentence in extracted_predicates:
			predicate_entity = predicate_in_sentence.predicate
			postag_entity = predicate_entity.part_of_speech
			postagged_word = POSTaggedWord(predicate_entity.lemma, postag_entity.part_of_speech)

			if postagged_word in predicate_counter:
				predicate_counter[postagged_word] += 1

			if self.has_reached_size(predicate_counter[postagged_word]):
				predicate_counter.pop(postagged_word)

		return predicate_counter

	@staticmethod
	def _store_parsing(doc: ParsedText, sentence_entity: Sentence, parser_entity: Parser):
		if Parsing.get(sentence=sentence_entity, parser=parser_entity) is None:
			Parsing(sentence=sentence_entity, parser=parser_entity, binary=doc.to_bytes())

	def _store_extractions_by_predicates(
			self, doc: ParsedText, multi_word_extraction: MultiWordExtraction,
			extractor_entity: Extractor, sentence_entity: Sentence,
			encoder_entity: Encoder, argument_encoder: ArgumentEncoder,
			predicate_counter: Counter
	):
		for predicate_idx, extractions in multi_word_extraction.extractions_per_idx.items():
			predicate = doc[predicate_idx]
			postagged_word = POSTaggedWord(predicate.lemma, predicate.pos)

			if postagged_word not in predicate_counter:
				continue

			predicate_counter[postagged_word] += 1

			verb = self.verb_translator.translate(predicate.lemma, predicate.pos)
			predicate_entity = get_predicate(verb, predicate.pos, predicate.lemma, generate_missing=True)
			predicate_in_sentence = get_predicate_in_sentence(sentence_entity, predicate_entity, predicate.i, generate_missing=True)

			combined_extraction = combine_extractions(extractions, safe_combine=False)
			insert_encoded_arguments(combined_extraction.args, extractor_entity, predicate_in_sentence, encoder_entity, argument_encoder)

			if self.has_reached_size(predicate_counter[postagged_word]):
				predicate_counter.pop(postagged_word)

	@db_session
	def _store_encoded_extractions(self, db_communicator: SQLiteCommunicator, docs: Iterator[ParsedText]):
		parser_entity = get_parser(
			engine_by_parser[type(self.dependency_parser)], self.dependency_parser.name, generate_missing=True)
		extractor_entity = get_extractor(EXTRACTORS_CONFIG.EXTRACTOR, parser_entity, generate_missing=True)
		encoder_entity = get_encoder(self.model_name, self.encoding_level, parser_entity, generate_missing=True)

		predicate_counter = self._get_predicate_counter(extractor_entity)

		loop_status = tqdm(docs, leave=False)
		for doc in loop_status:
			limited_idxs = self._get_potential_idxs(doc, extractor_entity, predicate_counter)
			multi_word_extraction = timeit(self.args_extractor.extract_multiword)(doc, limited_idxs=limited_idxs)
			if len(multi_word_extraction.extractions_per_idx) == 0:
				continue

			sentence_entity = timeit(get_sentence)(doc.text, generate_missing=True)
			arg_encoder = ENCODER_BY_LEVEL.get(self.encoding_level)(self.tokenizer, self.model, self.device, doc)

			timeit(self._store_parsing)(doc, sentence_entity, parser_entity)
			timeit(self._store_extractions_by_predicates)(
				doc, multi_word_extraction, extractor_entity, sentence_entity,
				encoder_entity, arg_encoder, predicate_counter)

			timeit(db_communicator.commit)()
			if len(predicate_counter) == 0:
				break

			loop_status.set_description(f"Missing: {len(predicate_counter)}")

	def append_dataset(self, out_dataset_path):
		in_parsed_dataset = self._load_parsed_dataset(self.in_dataset_path)

		create_db = not self.is_dataset_exist(out_dataset_path)
		db_communicator = SQLiteCommunicator(encoded_extractions_db, out_dataset_path, create_db=create_db)
		db_communicator.generate_mapping()

		self._store_encoded_extractions(db_communicator, in_parsed_dataset)
		db_communicator.disconnect()

	def create_dataset(self, out_dataset_path):
		if self.is_dataset_exist(out_dataset_path):
			os.remove(out_dataset_path)

		self.append_dataset(out_dataset_path)
