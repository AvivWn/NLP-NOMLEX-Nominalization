from typing import Optional

from tqdm import tqdm
from pony.orm import db_session

from yet_another_verb.arguments_extractor.args_extractor import ArgsExtractor
from yet_another_verb.arguments_extractor.extraction.utils.combination import combine_extractions
from yet_another_verb.data_handling.dataset_creator import DatasetCreator
from yet_another_verb.data_handling.db.communicators.sqlite_communicator import SQLiteCommunicator
from yet_another_verb.data_handling.db.encoded_extractions.queries import get_extractor, get_encoder, \
	get_predicate_in_sentence, get_predicate, get_parser, insert_encoded_arguments, get_extracted_indices_in_sentence
from yet_another_verb.data_handling.db.encoded_extractions.structure import encoded_extractions_db, \
	Parser, Parsing, Sentence, Extractor
from yet_another_verb.dependency_parsing import engine_by_parser
from yet_another_verb.dependency_parsing.dependency_parser.dependency_parser import DependencyParser
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText
from yet_another_verb.sentence_encoding.argument_encoding.encoding_level import EncodingLevel, encoder_by_level
from yet_another_verb.sentence_encoding.encoder import Encoder
from yet_another_verb.sentence_encoding.frameworks import framework_by_encoder
from yet_another_verb.word_to_verb.verb_translator import VerbTranslator
from yet_another_verb.configuration import EXTRACTORS_CONFIG
from yet_another_verb.utils.debug_utils import timeit


class EncodedExtractionsExpander(DatasetCreator):
	def __init__(
			self,
			dependency_parser: DependencyParser,
			args_extractor: ArgsExtractor,
			verb_translator: VerbTranslator,
			encoder: Encoder, encoding_level: EncodingLevel,
			dataset_size=None, **kwargs
	):
		super().__init__(dataset_size)

		self.dependency_parser = dependency_parser
		self.args_extractor = args_extractor
		self.verb_translator = verb_translator
		self.encoder = encoder
		self.encoding_level = encoding_level

	def _get_parsed_text(self, sentence_entity: Sentence, parser_entity) -> ParsedText:
		doc = Parsing.get(sentence=sentence_entity, parser=parser_entity)
		assert doc is not None
		return self.dependency_parser.from_bytes(doc.binary)

	def _expand_parsings(self, sentence_entity: Sentence, parser_entity: Parser) -> Optional[ParsedText]:
		if Parsing.get(sentence=sentence_entity, parser=parser_entity) is None:
			doc = self.dependency_parser(sentence_entity.text)
			Parsing(sentence=sentence_entity, parser=parser_entity, binary=doc.to_bytes())
			return doc

	def _expand_extractions(
			self, doc: ParsedText, sentence_entity: Sentence, extractor_entity: Extractor, encoder_entity: Encoder):
		# skip extracted sentences
		extracted_predicates = get_extracted_indices_in_sentence(sentence_entity, extractor_entity)
		if len(extracted_predicates) > 0:
			return

		arg_encoder = encoder_by_level.get(self.encoding_level)(parsed_text=doc, encoder=self.encoder)

		multi_word_extraction = self.args_extractor.extract_multiword(doc)
		for predicate_idx, extractions in multi_word_extraction.extractions_per_idx.items():
			predicate = doc[predicate_idx]
			verb = self.verb_translator.translate(predicate.lemma, predicate.pos)
			predicate_entity = get_predicate(verb, predicate.pos, predicate.lemma, generate_missing=True)
			predicate_in_sentence = get_predicate_in_sentence(sentence_entity, predicate_entity, predicate.i, generate_missing=True)

			combined_extraction = combine_extractions(extractions, safe_combine=False)
			insert_encoded_arguments(combined_extraction.args, extractor_entity, predicate_in_sentence, encoder_entity, arg_encoder)

	@db_session
	def _expand_dataset(self, db_communicator: SQLiteCommunicator):
		parser_entity = get_parser(
			engine_by_parser[type(self.dependency_parser)], self.dependency_parser.name, generate_missing=True)
		extractor_entity = get_extractor(EXTRACTORS_CONFIG.EXTRACTOR, parser_entity, generate_missing=True)
		encoder_entity = get_encoder(
			framework_by_encoder[type(self.encoder)], self.encoder.name, self.encoding_level,
			parser_entity, generate_missing=True)

		all_sentences = Sentence.select()
		for sentence_entity in tqdm(all_sentences, leave=False):
			doc = timeit(self._expand_parsings)(sentence_entity, parser_entity)
			doc = self._get_parsed_text(sentence_entity, parser_entity) if doc is None else doc
			timeit(self._expand_extractions)(doc, sentence_entity, extractor_entity, encoder_entity)
			db_communicator.commit()

	def append_dataset(self, out_dataset_path):
		db_communicator = SQLiteCommunicator(encoded_extractions_db, out_dataset_path, create_db=False)
		db_communicator.generate_mapping()

		self._expand_dataset(db_communicator)
		db_communicator.disconnect()

	def create_dataset(self, out_dataset_path):
		raise NotImplementedError()
