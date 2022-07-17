from collections import Counter
from os.path import isdir, join
from os import listdir
from typing import Iterator, List, Optional
from itertools import chain
import pickle

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from pony.orm import db_session

from yet_another_verb.arguments_extractor.args_extractor import ArgsExtractor
from yet_another_verb.arguments_extractor.extraction import Extractions, MultiWordExtraction
from yet_another_verb.data_handling import ParsedBinFileHandler
from yet_another_verb.data_handling.dataset_creator import DatasetCreator
from yet_another_verb.data_handling.db.communicators.sqlite_communicator import SQLiteCommunicator
from yet_another_verb.data_handling.db.encoded_extractions.queries import get_extractor, get_model, get_sentence, \
	get_predicate_in_sentence, get_predicate, get_extracted_predicates, get_extracted_idxs_in_sentence, get_parser
from yet_another_verb.data_handling.db.encoded_extractions.structure import encoded_extractions_db, Extraction, \
	Encoding, Parser, Parsing, Sentence, Model, Extractor
from yet_another_verb.dependency_parsing.dependency_parser.dependency_parser import DependencyParser
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText
from yet_another_verb.dependency_parsing.dependency_parser.parsed_word import ParsedWord
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
			limited_pos: List[POSTag] = None,
			limited_words: List[str] = None,
			dataset_size=None, **kwargs
	):
		super().__init__(dataset_size)
		self.in_dataset_path = in_dataset_path

		self.dependency_parser = dependency_parser
		self.args_extractor = args_extractor
		self.verb_translator = verb_translator

		self.model_name = model_name
		self.device = device
		self.tokenizer = AutoTokenizer.from_pretrained(model_name)
		self.model = AutoModel.from_pretrained(model_name).to(self.device)
		self.model.eval()

		self.limited_pos = limited_pos
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

		if self.limited_pos is not None and postagged_word.postag not in self.limited_pos:
			return False

		if predicate_counter is not None and postagged_word not in predicate_counter:
			return False

		return True

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

		return predicate_counter

	def _get_sentence_encoding(self, sentence: str) -> torch.Tensor:
		tokenized = self.tokenizer(sentence.split(), return_tensors="pt", is_split_into_words=True)
		tokenized = tokenized.to(self.device)

		with torch.no_grad():
			return self.model(**tokenized)[0][0].cpu()

	def _store_encoding(self, doc: ParsedText, sentence_entity: Sentence, model_entity: Model):
		if Encoding.get(sentence=sentence_entity, model=model_entity) is None:
			encoding = self._get_sentence_encoding(doc.tokenized_text)
			Encoding(sentence=sentence_entity, model=model_entity, binary=pickle.dumps(encoding))

	@staticmethod
	def _store_parsing(doc: ParsedText, sentence_entity: Sentence, parser_entity: Parser):
		if Parsing.get(sentence=sentence_entity, parser=parser_entity) is None:
			Parsing(sentence=sentence_entity, parser=parser_entity, binary=doc.to_bytes())

	def _store_extractions(
			self, extractions: Extractions, words: List[str], predicate: ParsedWord,
			extractor_entity: Extractor, sentence_entity: Sentence):
		assert len(extractions) > 0

		verb = self.verb_translator.translate(predicate.lemma, predicate.pos)
		predicate_entity = get_predicate(verb, predicate.pos, predicate.lemma, generate_missing=True)
		predicate_in_sentence = get_predicate_in_sentence(sentence_entity, predicate_entity, predicate.i, generate_missing=True)

		if Extraction.get(predicate_in_sentence=predicate_in_sentence, extractor=extractor_entity) is None:
			for extraction in extractions:
				extraction.words = words

			Extraction(
				predicate_in_sentence=predicate_in_sentence, extractor=extractor_entity,
				binary=pickle.dumps(extractions))

	def _store_extractions_by_predicates(
			self, doc: ParsedText, multi_word_extraction: MultiWordExtraction,
			extractor_entity: Extractor, sentence_entity: Sentence,
			predicate_counter: Counter):
		for predicate_idx, extractions in multi_word_extraction.extractions_per_idx.items():
			predicate = doc[predicate_idx]
			postagged_word = POSTaggedWord(predicate.lemma, predicate.pos)

			if postagged_word not in predicate_counter:
				continue

			predicate_counter[postagged_word] += 1

			self._store_extractions(extractions, doc.words, predicate, extractor_entity, sentence_entity)
			if predicate_counter[postagged_word] >= self.dataset_size:
				predicate_counter.pop(postagged_word)

	@db_session
	def _store_encoded_extractions(self, db_communicator: SQLiteCommunicator, docs: Iterator[ParsedText]):
		extractor_entity = get_extractor(EXTRACTORS_CONFIG.EXTRACTOR, generate_missing=True)
		model_entity = get_model(self.model_name, generate_missing=True)
		parser_entity = get_parser(
			engine_by_parser[type(self.dependency_parser)], self.dependency_parser.name, generate_missing=True)

		predicate_counter = self._get_predicate_counter(extractor_entity)

		loop_status = tqdm(docs, leave=False)
		for doc in loop_status:
			limited_idxs = {w.i for w in doc if self._is_relevant_predicate(POSTaggedWord(w.lemma, w.pos), predicate_counter)}
			if len(limited_idxs) == 0:
				continue

			sentence_entity = timeit(get_sentence)(doc.tokenized_text, generate_missing=True)
			extracted_idxs = get_extracted_idxs_in_sentence(sentence_entity, extractor_entity)
			limited_idxs = limited_idxs.difference(extracted_idxs)

			multi_word_extraction = timeit(self.args_extractor.extract_multiword)(doc, limited_idxs=limited_idxs)
			if len(multi_word_extraction.extractions_per_idx) == 0:
				continue

			timeit(self._store_encoding)(doc, sentence_entity, model_entity)
			timeit(self._store_parsing)(doc, sentence_entity, parser_entity)
			timeit(self._store_extractions_by_predicates)(
				doc, multi_word_extraction, extractor_entity, sentence_entity, predicate_counter)

			timeit(db_communicator.commit)()
			if len(predicate_counter) == 0:
				break

			loop_status.set_description(f"Missing: {len(predicate_counter)}")

	def create_dataset(self, out_dataset_path):
		in_parsed_dataset = self._load_parsed_dataset(self.in_dataset_path)

		db_communicator = SQLiteCommunicator(encoded_extractions_db, out_dataset_path)
		db_communicator.generate_mapping()

		self._store_encoded_extractions(db_communicator, in_parsed_dataset)
		db_communicator.disconnect()
