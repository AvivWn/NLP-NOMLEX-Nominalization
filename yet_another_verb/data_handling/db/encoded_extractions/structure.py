from pony.orm import *

_db = Database()
encoded_extractions_db = _db

T_EXTRACTOR = 'Extractor'
T_MODEl = 'Model'
T_PARSER = 'Parser'
T_VERB = 'VERB'
T_PART_OF_SPEECH = 'PartOfSpeech'
T_PREDICATE = 'Predicate'
T_SENTENCE = 'Sentence'
T_PREDICATE_IN_SENTENCE = 'PredicateInSentence'
T_EXTRACTION = 'Extraction'
T_ENCODING = 'Encoding'
T_PARSING = 'Parsing'


class Extractor(_db.Entity):
	id = PrimaryKey(int, auto=True)
	extractor = Required(str, unique=True)
	extractions = Set(T_EXTRACTION)


class Model(_db.Entity):
	id = PrimaryKey(int, auto=True)
	model = Required(str, unique=True)
	encodings = Set(T_ENCODING)


class Parser(_db.Entity):
	engine = Required(str)
	parser = Required(str)
	parsings = Set(T_PARSING)
	PrimaryKey(engine, parser)


class Sentence(_db.Entity):
	id = PrimaryKey(int, auto=True)
	text = Required(str)
	predicates = Set(T_PREDICATE_IN_SENTENCE)
	encodings = Set(T_ENCODING)
	parsings = Set(T_PARSING)


class Verb(_db.Entity):
	id = PrimaryKey(int, auto=True)
	lemma = Required(str, unique=True)
	predicates = Set(T_PREDICATE)


class PartOfSpeech(_db.Entity):
	id = PrimaryKey(int, auto=True)
	part_of_speech = Required(str, unique=True)
	predicates = Set(T_PREDICATE)


class Predicate(_db.Entity):
	verb = Required(Verb)
	part_of_speech = Required(PartOfSpeech)
	lemma = Required(str)
	predicates_in_sentences = Set(T_PREDICATE_IN_SENTENCE)
	PrimaryKey(verb, part_of_speech, lemma)


class PredicateInSentence(_db.Entity):
	sentence = Required(Sentence)
	predicate = Required(Predicate)
	word_index = Required(int)
	extractions = Set(T_EXTRACTION)
	PrimaryKey(sentence, predicate, word_index)


class Extraction(_db.Entity):
	predicate_in_sentence = Required(PredicateInSentence)
	extractor = Required(Extractor)
	binary = Required(bytes)
	PrimaryKey(predicate_in_sentence, extractor)


class Encoding(_db.Entity):
	sentence = Required(Sentence)
	model = Required(Model)
	binary = Required(bytes)
	PrimaryKey(sentence, model)


class Parsing(_db.Entity):
	sentence = Required(Sentence)
	parser = Required(Parser)
	binary = Required(bytes)
	PrimaryKey(sentence, parser)
