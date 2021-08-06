from enum import Enum


class LexiconType(str, Enum):
	VERB = "verb"
	NOUN = "noun"


LEXICON_TYPES = [t for t in LexiconType]
