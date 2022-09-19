from typing import Optional, Type, Union, List, Iterator

from pony.orm.core import Entity

from yet_another_verb.data_handling.db.encoded_extractions.structure import Extractor, \
	Model, Verb, PartOfSpeech, Predicate, Sentence, PredicateInSentence, Parser, Encoding, Extraction, Parsing


def get_entity_by_params(entity_type: Type[Entity], generate_missing, **kwargs) -> Optional[Entity]:
	entity = entity_type.get(**kwargs)

	if entity is None and generate_missing:
		entity = entity_type(**kwargs)

	return entity


def get_extractor(extractor: str, generate_missing=False) -> Optional[Extractor]:
	return get_entity_by_params(Extractor, generate_missing, extractor=extractor)


def get_model(model: str, generate_missing=False) -> Optional[Model]:
	return get_entity_by_params(Model, generate_missing, model=model)


def get_parser(engine: str, parser: str, generate_missing=False) -> Optional[Model]:
	return get_entity_by_params(Parser, generate_missing, engine=engine, parser=parser)


def get_sentence(text: str, generate_missing=False) -> Optional[Sentence]:
	return get_entity_by_params(Sentence, generate_missing, text=text)


def get_verb(lemma: str, generate_missing=False) -> Optional[Verb]:
	return get_entity_by_params(Verb, generate_missing, lemma=lemma)


def get_part_of_speech(pos: str, generate_missing=False) -> Optional[PartOfSpeech]:
	return get_entity_by_params(PartOfSpeech, generate_missing, part_of_speech=pos)


def get_predicate(
		verb: Union[Verb, str], pos: Union[PartOfSpeech, str],
		lemma: str, generate_missing=False) -> Optional[Predicate]:
	verb = verb if isinstance(verb, Verb) else get_verb(verb, generate_missing)
	pos = pos if isinstance(pos, PartOfSpeech) else get_part_of_speech(pos, generate_missing)
	return get_entity_by_params(Predicate, generate_missing, verb=verb, part_of_speech=pos, lemma=lemma)


def get_predicate_in_sentence(
		sentence: Sentence, predicate: Predicate,
		word_index: int, generate_missing=False) -> Optional[PredicateInSentence]:
	return get_entity_by_params(
		PredicateInSentence, generate_missing,
		sentence=sentence, predicate=predicate, word_index=word_index)


def get_extracted_idxs_in_sentence(sentence: Sentence, extractor: Extractor) -> List[int]:
	extracted_idxs = []

	for predicate_in_sentence in sentence.predicates:
		relevant_extractions = [e for e in predicate_in_sentence.extractions if e.extractor == extractor]
		if len(relevant_extractions) > 0:
			extracted_idxs.append(predicate_in_sentence.word_index)

	return extracted_idxs


def get_extracted_predicates(extractor: Extractor) -> List[PredicateInSentence]:
	predicates = []

	for extraction in extractor.extractions:
		predicates.append(extraction.predicate_in_sentence)

	return predicates


def get_limited_predicates(verb: str, postag: str) -> List[Predicate]:
	verb_entity = get_verb(verb)
	postag_entity = get_part_of_speech(postag)

	if verb_entity is None or postag_entity is None:
		return []

	return [p for p in verb_entity.predicates if p.part_of_speech == postag_entity]


def get_limited_encodings(sentence: Union[str, Sentence], model: Union[str, Model]) -> List[Encoding]:
	model_entity = model if isinstance(model, Model) else get_model(model)
	sentence_entity = sentence if isinstance(sentence, Sentence) else get_sentence(sentence)

	if model_entity is None or sentence_entity is None:
		return []

	return sentence_entity.encodings.select(lambda enc: enc.model == model_entity)


def get_limited_extractions(verb: str, postag: str, extractor: str) -> Iterator[Extraction]:
	extractor_entity = get_extractor(extractor)

	if extractor_entity is None:
		return []

	predicates = get_limited_predicates(verb, postag)

	for predicate in predicates:
		for predicate_in_sentence in predicate.predicates_in_sentences:
			extractions = [ext for ext in predicate_in_sentence.extractions if ext.extractor == extractor_entity]
			for ext in extractions:
				yield ext


def get_limited_parsings(sentence: Union[str, Sentence], engine: str, parser: str) -> List[Parsing]:
	sentence_entity = sentence if isinstance(sentence, Sentence) else get_sentence(sentence)
	parser_entity = get_parser(engine, parser)

	if sentence_entity is None or parser_entity is None:
		return []

	return [pars for pars in sentence_entity.parsings if pars.parser == parser_entity]
