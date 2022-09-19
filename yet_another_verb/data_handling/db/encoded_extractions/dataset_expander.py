import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from pony.orm import db_session

from yet_another_verb.arguments_extractor.args_extractor import ArgsExtractor
from yet_another_verb.data_handling import TorchBytesHandler
from yet_another_verb.data_handling.dataset_creator import DatasetCreator
from yet_another_verb.data_handling.db.communicators.sqlite_communicator import SQLiteCommunicator
from yet_another_verb.data_handling.db.encoded_extractions.queries import get_extractor, get_model, \
	get_predicate_in_sentence, get_predicate, get_parser, generate_extraction
from yet_another_verb.data_handling.db.encoded_extractions.structure import encoded_extractions_db, \
	Encoding, Parser, Parsing, Sentence, Model, Extractor
from yet_another_verb.dependency_parsing.dependency_parser.dependency_parser import DependencyParser
from yet_another_verb.dependency_parsing import engine_by_parser
from yet_another_verb.word_to_verb.verb_translator import VerbTranslator
from yet_another_verb.configuration import EXTRACTORS_CONFIG
from yet_another_verb.utils.debug_utils import timeit


class EncodedExtractionsExpander(DatasetCreator):
	def __init__(
			self,
			dependency_parser: DependencyParser,
			args_extractor: ArgsExtractor,
			verb_translator: VerbTranslator,
			model_name: str, device: str,
			dataset_size=None, **kwargs
	):
		super().__init__(dataset_size)

		self.dependency_parser = dependency_parser
		self.args_extractor = args_extractor
		self.verb_translator = verb_translator

		self.model_name = model_name
		self.device = device
		self.tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
		self.model = AutoModel.from_pretrained(model_name).to(self.device)
		self.model.eval()

	def _get_sentence_encoding(self, sentence: str) -> torch.Tensor:
		tokenized = self.tokenizer(sentence.split(), return_tensors="pt", is_split_into_words=True, add_special_tokens=True)
		tokenized = tokenized.to(self.device)

		with torch.no_grad():
			return self.model(**tokenized)[0][0].cpu()

	def _expand_encodings(self, sentence_entity: Sentence, model_entity: Model):
		if Encoding.get(sentence=sentence_entity, model=model_entity) is None:
			encoding = self._get_sentence_encoding(sentence_entity.text)
			Encoding(sentence=sentence_entity, model=model_entity, binary=TorchBytesHandler.saves(encoding))

	def _expand_parsings(self, sentence_entity: Sentence, parser_entity: Parser):
		if Parsing.get(sentence=sentence_entity, parser=parser_entity) is None:
			doc = self.dependency_parser(sentence_entity.text.split())
			assert doc.tokenized_text == sentence_entity.text  # The tokenization is not assumed to change...
			Parsing(sentence=sentence_entity, parser=parser_entity, binary=doc.to_bytes())

	def _expand_extractions(
			self, sentence_entity: Sentence, extractor_entity: Extractor, parser_entity: Parser):
		# skip extracted sentences
		extracted_predicates = sentence_entity.predicates.select(
			lambda p: len(p.extractions.select(lambda e: e.extractor == extractor_entity)) > 0)
		if len(extracted_predicates) > 0:
			return

		doc = Parsing.get(sentence=sentence_entity, parser=parser_entity)
		assert doc is not None
		parsed_text = self.dependency_parser.from_bytes(doc.binary)

		multi_word_extraction = self.args_extractor.extract_multiword(parsed_text)
		for predicate_idx, extractions in multi_word_extraction.extractions_per_idx.items():
			predicate = parsed_text[predicate_idx]
			verb = self.verb_translator.translate(predicate.lemma, predicate.pos)
			predicate_entity = get_predicate(verb, predicate.pos, predicate.lemma, generate_missing=True)
			predicate_in_sentence = get_predicate_in_sentence(sentence_entity, predicate_entity, predicate.i, generate_missing=True)
			generate_extraction(extractions, doc.words, predicate_in_sentence, extractor_entity)

	@db_session
	def _expand_dataset(self, db_communicator: SQLiteCommunicator):
		extractor_entity = get_extractor(EXTRACTORS_CONFIG.EXTRACTOR, generate_missing=True)
		model_entity = get_model(self.model_name, generate_missing=True)
		parser_entity = get_parser(
			engine_by_parser[type(self.dependency_parser)], self.dependency_parser.name, generate_missing=True)

		all_sentences = Sentence.select()
		for sentence_entity in tqdm(all_sentences, leave=False):
			timeit(self._expand_encodings)(sentence_entity, model_entity)
			timeit(self._expand_parsings)(sentence_entity, parser_entity)
			timeit(self._expand_extractions)(sentence_entity, extractor_entity, parser_entity)
			db_communicator.commit()

	def append_dataset(self, out_dataset_path):
		db_communicator = SQLiteCommunicator(encoded_extractions_db, out_dataset_path, create_db=False)
		db_communicator.generate_mapping()

		self._expand_dataset(db_communicator)
		db_communicator.disconnect()

	def create_dataset(self, out_dataset_path):
		raise NotImplementedError()
