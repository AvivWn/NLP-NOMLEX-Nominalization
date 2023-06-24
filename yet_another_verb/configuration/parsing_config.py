from yet_another_verb.dependency_parsing.spacy.spacy_parser import SpacyParser

SPACY_ENGINE = "spacy"


class ParsingConfig:
	def __init__(
			self,
			parsing_engine=SPACY_ENGINE,
			parser_name="en_ud_model_lg"
	):
		self.PARSING_ENGINE = parsing_engine
		self.PARSER_NAME = parser_name


PARSING_CONFIG = ParsingConfig()


PARSER_BY_ENGINE = {
	SPACY_ENGINE: SpacyParser
}
ENGINE_BY_PARSER = {v: k for k, v in PARSER_BY_ENGINE.items()}
