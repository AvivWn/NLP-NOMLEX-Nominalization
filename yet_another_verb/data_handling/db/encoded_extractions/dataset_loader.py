from time import time
from typing import List
from collections import Counter, defaultdict

from pony.orm import db_session
from tqdm import tqdm

from yet_another_verb.arguments_extractor.extraction import ExtractedArgument, Extraction, ArgumentType
from yet_another_verb.arguments_extractor.extraction.extraction import EncodedExtraction
from yet_another_verb.data_handling import TorchBytesHandler
from yet_another_verb.data_handling.db.communicators.sqlite_communicator import SQLiteCommunicator
from yet_another_verb.data_handling.db.encoded_extractions.encodings import EncodingLevel
from yet_another_verb.data_handling.db.encoded_extractions.queries import get_limited_encodings, \
	get_extracted_predicates, get_extractor, get_encoder, get_parser, get_limited_parsings
from yet_another_verb.data_handling.db.encoded_extractions.structure import ExtractedArgument as DBExtractedArgument
from yet_another_verb.data_handling.db.encoded_extractions.structure import encoded_extractions_db, Encoder
from yet_another_verb.dependency_parsing import POSTaggedWord
from yet_another_verb.factories.dependency_parser_factory import DependencyParserFactory


class EncodedExtractionsLoader:
	def __init__(self, dataset_path: str, parsing_engine: str, parser_name: str):
		self.db_communicator = SQLiteCommunicator(encoded_extractions_db, dataset_path, create_db=False)
		self.db_communicator.generate_mapping()

		self.parsing_engine = parsing_engine
		self.parser_name = parser_name
		self.parser = DependencyParserFactory(parsing_engine=parsing_engine, parser_name=parser_name)()

		with db_session:
			self.parser_entity = get_parser(self.parsing_engine, self.parser_name)

	@staticmethod
	def _agg_args_by_predicate(extracted_arg_entites):
		arguments_by_predicates = defaultdict(list)

		for extracted_arg_entity in tqdm(extracted_arg_entites, leave=False):
			arg_entity = extracted_arg_entity.argument
			predicate_in_sentence = arg_entity.predicate_in_sentence
			arguments_by_predicates[predicate_in_sentence].append(extracted_arg_entity)

		return arguments_by_predicates

	@staticmethod
	def _get_encoded_args(extracted_arg_entities: List[DBExtractedArgument], encoder_entity: Encoder):
		encoded_args = {}

		for extracted_arg_entity in extracted_arg_entities:
			encodings = list(get_limited_encodings(extracted_arg_entity.argument, encoder_entity))
			assert len(encodings) > 0
			encoded_args[ArgumentType(extracted_arg_entity.argument_type.argument_type)] = TorchBytesHandler.loads(encodings[0].binary)

		return encoded_args

	@db_session
	def get_encoded_extractions(
			self, extractor: str, encoding_model: str, encoding_level: EncodingLevel
	) -> List[EncodedExtraction]:
		extractor_entity = get_extractor(extractor, self.parser_entity)
		encoder_entity = get_encoder(encoding_model, encoding_level, self.parser_entity)

		if extractor_entity is None or encoder_entity is None:
			return []

		args_by_predicate = self._agg_args_by_predicate(extractor_entity.extracted_arguments)

		encoded_extractions = []
		for predicate_in_sentence, extracted_arg_entities in tqdm(args_by_predicate.items(), False):
			a = time()
			parsings = list(get_limited_parsings(predicate_in_sentence.sentence, self.parser_entity))
			assert len(parsings) > 0
			print("1", time() - a)

			a = time()
			extraction = Extraction(
				words=self.parser.from_bytes(parsings[0].binary),
				predicate_idx=predicate_in_sentence.word_idx,
				predicate_lemma=predicate_in_sentence.predicate.lemma,
				args=[ExtractedArgument(
					arg_idxs=list(range(e.argument.start_idx, e.argument.end_idx + 1)),
					arg_type=ArgumentType(e.argument_type.argument_type)) for e in extracted_arg_entities])
			print("2", time() - a)

			a = time()
			encoded_args = self._get_encoded_args(extracted_arg_entities, encoder_entity)
			encoded_extractions.append(EncodedExtraction(extraction=extraction, encoded_args=encoded_args))
			print("3", time() - a)

		return encoded_extractions

	@db_session
	def get_predicate_counts(self, extractor: str) -> Counter:
		extractor_entity = get_extractor(extractor, self.parser_entity)
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


if __name__ == "__main__":
	db_path = "/home/nlp/avivwn/thesis/data/wiki40b/encoded-extractions/limited-words/part00.db"
	with EncodedExtractionsLoader(db_path, "spacy", "en_ud_model_lg") as loader:
		result = loader.get_encoded_extractions("nomlex", "bert-base-uncased", EncodingLevel.HEAD_IDX)
		print(len(result))
