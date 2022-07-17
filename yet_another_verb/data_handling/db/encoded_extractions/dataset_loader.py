from typing import Optional, List
from collections import Counter

import torch
from pony.orm import db_session

from yet_another_verb.arguments_extractor.extraction import Extractions
from yet_another_verb.data_handling.binary.pkl_handler import PKLHandler
from yet_another_verb.data_handling.db.communicators.sqlite_communicator import SQLiteCommunicator
from yet_another_verb.data_handling.db.encoded_extractions.queries import get_limited_extractions, \
	get_limited_parsings, get_limited_encodings, get_extracted_predicates, get_extractor
from yet_another_verb.data_handling.db.encoded_extractions.structure import encoded_extractions_db
from yet_another_verb.dependency_parsing import POSTag, POSTaggedWord
from yet_another_verb.factories.dependency_parser_factory import DependencyParserFactory


class EncodedExtractionsLoader:
	def __init__(self, dataset_path: str):
		self.db_communicator = SQLiteCommunicator(encoded_extractions_db, dataset_path)
		self.db_communicator.generate_mapping()

	@db_session
	def get_encoding(self, sentence: str, model: str) -> Optional[torch.Tensor]:
		encodings = get_limited_encodings(sentence, model)

		if len(encodings) == 0:
			return None

		return PKLHandler.loads(encodings[0])

	@db_session
	def get_extractions(
			self, verb: str, postag: POSTag,
			extractor: str, engine: str, parser: str,
			amount: int = None) -> List[Extractions]:
		extraction_entities = get_limited_extractions(verb, postag, extractor)
		dependency_parser = DependencyParserFactory(parsing_engine=engine, parser_name=parser)()

		extractions = []
		for ext_entity in extraction_entities:
			parsings = get_limited_parsings(ext_entity.predicate_in_sentence.sentence, engine, parser)
			if len(parsings) == 0:
				continue

			parsing = dependency_parser.from_bytes(parsings[0].binary)
			multiple_exts = PKLHandler.loads(ext_entity.binary)

			for ext in multiple_exts:
				ext.words = parsing

			if amount is None or len(extractions) < amount:
				extractions.append(multiple_exts)

		return extractions

	@db_session
	def get_predicate_counts(self, extractor: str) -> Counter:
		extractor_entity = get_extractor(extractor)
		extracted_predicates = get_extracted_predicates(extractor_entity)
		predicate_counter = Counter()
		for predicate_in_sentence in extracted_predicates:
			predicate_entity = predicate_in_sentence.predicate
			postag_entity = predicate_entity.part_of_speech
			postagged_word = POSTaggedWord(predicate_entity.lemma, postag_entity.part_of_speech)

			predicate_counter[postagged_word] += 1

		return predicate_counter
