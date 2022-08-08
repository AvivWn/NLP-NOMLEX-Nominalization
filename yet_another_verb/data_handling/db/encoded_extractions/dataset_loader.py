from typing import Optional, List, Union, Dict
from collections import Counter

import torch
from pony.orm import db_session
from tqdm import tqdm

from yet_another_verb.arguments_extractor.extraction import Extractions
from yet_another_verb.data_handling import TorchBytesHandler
from yet_another_verb.data_handling.bytes.pkl_bytes_handler import PKLBytesHandler
from yet_another_verb.data_handling.db.communicators.sqlite_communicator import SQLiteCommunicator
from yet_another_verb.data_handling.db.encoded_extractions.queries import get_limited_extractions, \
	get_limited_parsings, get_limited_encodings, get_extracted_predicates, get_extractor, get_model
from yet_another_verb.data_handling.db.encoded_extractions.structure import encoded_extractions_db, Extraction
from yet_another_verb.dependency_parsing import POSTag, POSTaggedWord
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText
from yet_another_verb.factories.dependency_parser_factory import DependencyParserFactory


class EncodedExtractionsLoader:
	def __init__(self, dataset_path: str, extractor: str, parsing_engine: str, parser_name: str):
		self.db_communicator = SQLiteCommunicator(encoded_extractions_db, dataset_path, create_db=False)
		self.db_communicator.generate_mapping()

		self.extractor = extractor
		self.parsing_engine = parsing_engine
		self.parser_name = parser_name
		self.parser = DependencyParserFactory(parsing_engine=parsing_engine, parser_name=parser_name)()

	@db_session
	def get_encoding(self, sentence: Union[ParsedText, str], model: str) -> Optional[torch.Tensor]:
		sentence = sentence.tokenized_text if isinstance(sentence, ParsedText) else sentence
		encodings = get_limited_encodings(sentence, model)

		if len(encodings) == 0:
			return None

		return TorchBytesHandler.loads(encodings[0].binary)

	@db_session
	def get_encodings_by_sentences(self, model: str) -> Dict[str, torch.Tensor]:
		model_entity = get_model(model)

		if model_entity is None:
			return {}

		encodings_by_sentences = {}
		encoding_entities = model_entity.encodings
		for enc_entity in tqdm(encoding_entities, leave=False):
			encodings_by_sentences[enc_entity.sentence.text] = TorchBytesHandler.loads(enc_entity.binary)

		return encodings_by_sentences

	def _get_raw_extraction(self, ext_entity: Extraction) -> Optional[Extractions]:
		sentence_entity = ext_entity.predicate_in_sentence.sentence
		parsings = get_limited_parsings(sentence_entity, self.parsing_engine, self.parser_name)
		if len(parsings) == 0:
			return None

		parsing = self.parser.from_bytes(parsings[0].binary)
		multiple_exts = PKLBytesHandler.loads(ext_entity.binary)

		for ext in multiple_exts:
			ext.words = parsing

		return multiple_exts

	@db_session
	def get_extractions(self, verb: str, postag: POSTag, amount: int = None) -> List[Extractions]:
		extraction_entities = get_limited_extractions(verb, postag, self.extractor)

		extractions = []
		for ext_entity in extraction_entities:
			raw_ext = self._get_raw_extraction(ext_entity)
			extractions.append(raw_ext)
			if amount is not None and len(extractions) >= amount:
				break

		return extractions

	@db_session
	def get_all_extractions(self) -> List[Extractions]:
		extractor_entity = get_extractor(self.extractor)

		if extractor_entity is None:
			return []

		extractions = []
		extraction_entities = extractor_entity.extractions
		for ext_entity in tqdm(extraction_entities, leave=False):
			raw_ext = self._get_raw_extraction(ext_entity)
			extractions.append(raw_ext)

		return extractions

	@db_session
	def get_predicate_counts(self) -> Counter:
		extractor_entity = get_extractor(self.extractor)
		extracted_predicates = get_extracted_predicates(extractor_entity)
		predicate_counter = Counter()
		for predicate_in_sentence in extracted_predicates:
			predicate_entity = predicate_in_sentence.predicate
			postag_entity = predicate_entity.part_of_speech
			postagged_word = POSTaggedWord(predicate_entity.lemma, postag_entity.part_of_speech)

			predicate_counter[postagged_word] += 1

		return predicate_counter

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_value, exc_traceback):
		self.db_communicator.__exit__(exc_type, exc_value, exc_traceback)
