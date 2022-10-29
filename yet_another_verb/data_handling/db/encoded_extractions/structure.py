from pony.orm import PrimaryKey, Required, Set, Database

_db = Database()
encoded_extractions_db = _db

T_EXTRACTOR = 'Extractor'
T_ENCODER = 'Encoder'
T_PREDICATE = 'Predicate'
T_PREDICATE_IN_SENTENCE = 'PredicateInSentence'
T_EXTRACTED_ARGUMENT = 'ExtractedArgument'
T_ARGUMENT = 'Argument'
T_ENCODING = 'Encoding'
T_PARSING = 'Parsing'


class Parser(_db.Entity):
	engine = Required(str)
	parser = Required(str)
	extractor = Set(T_EXTRACTOR)
	encoder = Set(T_ENCODER)
	parsings = Set(T_PARSING)
	PrimaryKey(engine, parser)


class Extractor(_db.Entity):
	extractor = Required(str)
	parser = Required(Parser)
	extracted_arguments = Set(T_EXTRACTED_ARGUMENT)
	PrimaryKey(extractor, parser)


class Encoder(_db.Entity):
	model = Required(str)
	encoding_level = Required(str)  # head-idx, start-idx, ...
	parser = Required(Parser)
	encodings = Set(T_ENCODING)
	PrimaryKey(model, encoding_level, parser)


class Sentence(_db.Entity):
	text = PrimaryKey(str)
	predicates = Set(T_PREDICATE_IN_SENTENCE)
	parsings = Set(T_PARSING)


class Verb(_db.Entity):
	lemma = PrimaryKey(str)
	predicates = Set(T_PREDICATE)


class PartOfSpeech(_db.Entity):
	part_of_speech = PrimaryKey(str)
	predicates = Set(T_PREDICATE)


class ArgumentType(_db.Entity):
	argument_type = PrimaryKey(str)
	extracted_arguments = Set(T_EXTRACTED_ARGUMENT)


class Predicate(_db.Entity):
	verb = Required(Verb)
	part_of_speech = Required(PartOfSpeech)
	lemma = Required(str)
	predicates_in_sentences = Set(T_PREDICATE_IN_SENTENCE)
	PrimaryKey(verb, part_of_speech, lemma)


class PredicateInSentence(_db.Entity):
	sentence = Required(Sentence)
	predicate = Required(Predicate)
	word_idx = Required(int)
	arguments = Set(T_ARGUMENT)
	PrimaryKey(sentence, word_idx, predicate)


class Argument(_db.Entity):
	predicate_in_sentence = Required(PredicateInSentence)
	start_idx = Required(int)
	end_idx = Required(int)
	extracted_arguments = Set(T_EXTRACTED_ARGUMENT)
	encodings = Set(T_ENCODING)
	PrimaryKey(predicate_in_sentence, start_idx, end_idx)


class ExtractedArgument(_db.Entity):
	argument = Required(Argument)
	extractor = Required(Extractor)
	argument_type = Required(ArgumentType)
	PrimaryKey(argument, extractor, argument_type)


class Encoding(_db.Entity):
	argument = Required(Argument)
	encoder = Required(Encoder)
	binary = Required(bytes)
	PrimaryKey(argument, encoder)


class Parsing(_db.Entity):
	sentence = Required(Sentence)
	parser = Required(Parser)
	binary = Required(bytes)
	PrimaryKey(sentence, parser)
