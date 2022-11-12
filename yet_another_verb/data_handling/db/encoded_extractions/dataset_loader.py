from collections import Counter, defaultdict
from typing import List

from pony.orm import db_session
from tqdm import tqdm

from yet_another_verb.arguments_extractor.extraction import ExtractedArgument, Extraction, ArgumentType
from yet_another_verb.arguments_extractor.extraction.extraction import Extractions
from yet_another_verb.arguments_extractor.extraction.words import Words
from yet_another_verb.data_handling import TorchBytesHandler, ExtractedBytesHandler
from yet_another_verb.data_handling.bytes.compressed.compressed_encoding import CompressedEncoding
from yet_another_verb.data_handling.bytes.compressed.compressed_parsed_text import CompressedParsedText
from yet_another_verb.data_handling.db.communicators.sqlite_communicator import SQLiteCommunicator
from yet_another_verb.sentence_encoding.argument_encoding.encoding_level import EncodingLevel
from yet_another_verb.data_handling.db.encoded_extractions.queries import get_limited_encodings, \
	get_extracted_predicates, get_extractor, get_encoder, get_parser, get_limited_parsings
from yet_another_verb.data_handling.db.encoded_extractions.structure import ExtractedArgument as DBExtractedArgument, \
	Parser
from yet_another_verb.data_handling.db.encoded_extractions.structure import encoded_extractions_db, Encoder
from yet_another_verb.dependency_parsing import POSTaggedWord, POSTag
from yet_another_verb.factories.dependency_parser_factory import DependencyParserFactory
from yet_another_verb.sentence_encoding.encoding import Encoding


class EncodedExtractionsLoader:
	def __init__(self, dataset_path: str, parsing_engine: str, parser_name: str, keep_compressed: bool = False):
		self.db_communicator = SQLiteCommunicator(encoded_extractions_db, dataset_path, create_db=False)
		self.db_communicator.generate_mapping()

		self.parsing_engine = parsing_engine
		self.parser_name = parser_name
		self.parser = DependencyParserFactory(parsing_engine=parsing_engine, parser_name=parser_name)()
		self.extracted_bytes_handler = ExtractedBytesHandler(self.parser)

		self.keep_compressed = keep_compressed

	@staticmethod
	def _agg_args_by_predicate(extracted_arg_entites):
		arguments_by_predicates = defaultdict(list)

		for extracted_arg_entity in tqdm(extracted_arg_entites, leave=False):
			arg_entity = extracted_arg_entity.argument
			predicate_in_sentence = arg_entity.predicate_in_sentence
			arguments_by_predicates[predicate_in_sentence].append(extracted_arg_entity)

		return arguments_by_predicates

	def _get_parsed_text(self, predicate_in_sentence, parser_entity: Parser) -> Words:
		parsings = list(get_limited_parsings(predicate_in_sentence.sentence, parser_entity))
		assert len(parsings) > 0
		parsing = parsings[0]

		if self.keep_compressed:
			return CompressedParsedText(
				bytes_data=parsing.binary,
				parsing_egnine=parser_entity.engine,
				parser_name=parser_entity.parser)
		else:
			return self.parser.from_bytes(parsing.binary)

	def _get_encoded_arg(self, extracted_arg_entity: DBExtractedArgument, encoder_entity: Encoder) -> Encoding:
		encodings = list(get_limited_encodings(extracted_arg_entity.argument, encoder_entity))
		assert len(encodings) > 0
		encoding = encodings[0]

		if self.keep_compressed:
			return CompressedEncoding(
				bytes_data=encoding.binary,
				encoding_framework=encoder_entity.framework,
				encoder_name=encoder_entity.encoder)
		else:
			return TorchBytesHandler.loads(encoding.binary)

	@db_session
	def get_encoded_extractions(
			self, extractor: str, encoding_framework: str, encoding_model: str, encoding_level: EncodingLevel,
			limited_postags: List[POSTag] = None
	) -> Extractions:
		parser_entity = get_parser(self.parsing_engine, self.parser_name)
		extractor_entity = get_extractor(extractor, parser_entity)
		encoder_entity = get_encoder(encoding_framework, encoding_model, encoding_level, parser_entity)

		if extractor_entity is None or encoder_entity is None:
			return []

		args_by_predicate = self._agg_args_by_predicate(extractor_entity.extracted_arguments)

		encoded_extractions = []
		for predicate_in_sentence, extracted_arg_entities in tqdm(list(args_by_predicate.items()), False):
			pos_entity = predicate_in_sentence.predicate.part_of_speech
			if limited_postags is not None and POSTag(pos_entity.part_of_speech) not in limited_postags:
				continue

			extracted_args = []
			for extracted_arg_entity in extracted_arg_entities:
				arg_entity = extracted_arg_entity.argument
				arg_type = extracted_arg_entity.argument_type.argument_type
				extracted_args.append(ExtractedArgument(
					arg_idxs=list(range(arg_entity.start_idx, arg_entity.end_idx + 1)),
					arg_type=ArgumentType(arg_type),
					encoding=self._get_encoded_arg(extracted_arg_entity, encoder_entity)))

			extraction = Extraction(
				words=self._get_parsed_text(predicate_in_sentence, parser_entity),
				predicate_idx=predicate_in_sentence.word_idx,
				predicate_lemma=predicate_in_sentence.predicate.lemma,
				args=extracted_args
			)

			encoded_extractions.append(extraction)

		return encoded_extractions

	@db_session
	def get_predicate_counts(self, extractor: str) -> Counter:
		parser_entity = get_parser(self.parsing_engine, self.parser_name)
		extractor_entity = get_extractor(extractor, parser_entity)
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
