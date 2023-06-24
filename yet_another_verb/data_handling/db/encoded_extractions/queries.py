from typing import Optional, Type, Union, List

from pony.orm.core import Entity

from yet_another_verb.arguments_extractor.extraction import ExtractedArgument
from yet_another_verb.data_handling import TorchBytesHandler
from yet_another_verb.sentence_encoding.argument_encoding.arg_encoder import ArgumentEncoder
from yet_another_verb.data_handling.db.encoded_extractions.structure import Extractor, Encoder, Verb, \
	PartOfSpeech, Predicate, Sentence, PredicateInSentence, Parser, Encoding, Argument, ArgumentType, \
	Parsing, ExtractedArgument as DBExtractedArgument


def get_entity_by_params(entity_type: Type[Entity], generate_missing, properties=None, **primary_keys) -> Optional[Entity]:
	properties = {} if properties is None else properties
	entity = entity_type.get(**primary_keys)

	if entity is None and generate_missing:
		entity = entity_type(**primary_keys, **properties)

	return entity


def get_parser(engine: str, parser: str, generate_missing=False) -> Optional[Parser]:
	return get_entity_by_params(Parser, generate_missing, engine=engine, parser=parser)


def get_extractor(extractor: str, parser: Parser, generate_missing=False) -> Optional[Extractor]:
	return get_entity_by_params(Extractor, generate_missing, extractor=extractor, parser=parser)


def get_encoder(framework: str, encoder: str, encoding_level: str, parser: Parser, generate_missing=False) -> Optional[Encoder]:
	return get_entity_by_params(Encoder, generate_missing, framework=framework, encoder=encoder, encoding_level=encoding_level, parser=parser)


def get_sentence(text: str, generate_missing=False) -> Optional[Sentence]:
	return get_entity_by_params(Sentence, generate_missing, text=text)


def get_verb(lemma: str, generate_missing=False) -> Optional[Verb]:
	return get_entity_by_params(Verb, generate_missing, lemma=lemma)


def get_part_of_speech(pos: str, generate_missing=False) -> Optional[PartOfSpeech]:
	return get_entity_by_params(PartOfSpeech, generate_missing, part_of_speech=pos)


def get_argument_type(argument_type: str, generate_missing=False) -> Optional[ArgumentType]:
	return get_entity_by_params(ArgumentType, generate_missing, argument_type=argument_type)


def get_predicate(
		verb: Union[Verb, str], pos: Union[PartOfSpeech, str],
		lemma: str, generate_missing=False) -> Optional[Predicate]:
	verb = verb if isinstance(verb, Verb) else get_verb(verb, generate_missing)
	pos = pos if isinstance(pos, PartOfSpeech) else get_part_of_speech(pos, generate_missing)
	return get_entity_by_params(Predicate, generate_missing, verb=verb, part_of_speech=pos, lemma=lemma)


def get_predicate_in_sentence(
		sentence: Sentence, predicate: Predicate,
		word_idx: int, generate_missing=False) -> Optional[PredicateInSentence]:
	return get_entity_by_params(
		PredicateInSentence, generate_missing,
		sentence=sentence, predicate=predicate, word_idx=word_idx)


def get_argument(
		extracted_arg: ExtractedArgument,
		predicate_in_sentence: PredicateInSentence, generate_missing=False) -> Optional[Argument]:
	return get_entity_by_params(
		Argument, generate_missing,
		predicate_in_sentence=predicate_in_sentence,
		start_idx=extracted_arg.start_idx, end_idx=extracted_arg.end_idx,
		head_idx=extracted_arg.head_idx
	)


def get_extracted_argument(
		arg: Argument, extractor: Extractor, arg_type: Union[str, ArgumentType],
		generate_missing=False) -> Optional[DBExtractedArgument]:
	arg_type = arg_type if isinstance(arg_type, ArgumentType) else get_argument_type(arg_type, generate_missing)
	return get_entity_by_params(
		DBExtractedArgument, generate_missing,
		argument=arg, extractor=extractor, argument_type=arg_type)


def get_extracted_args_in_sentence(sentence: Sentence, extractor: Extractor) -> List[DBExtractedArgument]:
	extracted_args = []

	for predicate_in_sentence in sentence.predicates:
		for arg in predicate_in_sentence.arguments:
			if any(e.extractor == extractor for e in arg.extracted_arguments):
				extracted_args.append(arg)

	return extracted_args


def get_extracted_indices_in_sentence(sentence: Sentence, extractor: Extractor) -> List[int]:
	extracted_indices = []

	for predicate_in_sentence in sentence.predicates:
		for arg in predicate_in_sentence.arguments:
			if any(e.extractor == extractor for e in arg.extracted_arguments):
				extracted_indices.append(predicate_in_sentence.word_idx)

	return extracted_indices


def get_extracted_predicates(extractor: Extractor) -> List[PredicateInSentence]:
	extracted_predicates = []

	for extracted_arg in extractor.extracted_arguments:
		extracted_predicates.append(extracted_arg.argument.predicate_in_sentence)

	return extracted_predicates


def get_limited_predicates(verb: str, postag: str) -> List[Predicate]:
	verb_entity = get_verb(verb)
	postag_entity = get_part_of_speech(postag)

	if verb_entity is None or postag_entity is None:
		return []

	return [p for p in verb_entity.predicates if p.part_of_speech == postag_entity]


def get_limited_encodings(argument: Argument, encoder: Encoder) -> List[Encoding]:
	return argument.encodings.select(lambda enc: enc.encoder == encoder)


def get_limited_parsings(sentence: Union[str, Sentence], parser: Parser) -> List[Parsing]:
	sentence_entity = sentence if isinstance(sentence, Sentence) else get_sentence(sentence)

	if sentence_entity is None:
		return []

	return [pars for pars in sentence_entity.parsings if pars.parser == parser]


def insert_encoding(argument_entity: Argument, encoder_entity: Encoder, arg_encoder: ArgumentEncoder):
	if Encoding.get(argument=argument_entity, encoder=encoder_entity) is None:
		extracted_arg = ExtractedArgument(
			start_idx=argument_entity.start_idx,
			end_idx=argument_entity.end_idx,
			head_idx=argument_entity.head_idx
		)

		argument_encoding = arg_encoder.encode(extracted_arg)
		binary_encoding = TorchBytesHandler.saves(argument_encoding)
		Encoding(argument=argument_entity, encoder=encoder_entity, binary=binary_encoding)


def insert_encoded_arguments(
		arguments: List[ExtractedArgument], extractor_entity: Extractor, predicate_in_sentence: PredicateInSentence,
		encoder_entity: Encoder, arg_encoder: ArgumentEncoder):
	for extracted_arg in arguments:
		argument_entity = get_argument(extracted_arg, predicate_in_sentence, generate_missing=True)
		get_extracted_argument(argument_entity, extractor_entity, extracted_arg.arg_type, generate_missing=True)

		insert_encoding(argument_entity, encoder_entity, arg_encoder)
