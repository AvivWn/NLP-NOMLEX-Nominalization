from typing import Optional, Type, Union, List

from pony.orm.core import Entity

from yet_another_verb.data_handling.db.encoded_extractions.structure import Extractor, \
	Model, Verb, PartOfSpeech, Predicate, Sentence, PredicateInSentence, Parser


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
