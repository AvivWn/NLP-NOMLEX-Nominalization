class ParsingConfig:
	def __init__(
			self,
			parsing_engine="spacy",
			parser_name="en_ud_model_lg"
	):
		self.PARSING_ENGINE = parsing_engine
		self.PARSER_NAME = parser_name


PARSING_CONFIG = ParsingConfig()
