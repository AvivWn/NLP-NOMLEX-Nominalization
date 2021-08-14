from yet_another_verb.dependency_parsing.spacy_en_ud.ud_parser import UDParser
from yet_another_verb.dependency_parsing.spacy_en_ud.ud_parsed_bin import UDParsedBin


class ParsingConfig:
	def __init__(
			self,
			parser_name="en_ud_model_lg"
	):
		self.PARSER_NAME = parser_name
		self.DEFAULT_PARSER_MAKER = lambda: UDParser(parser_name)
		self.DEFAULT_PARSED_BIN_MAKER = lambda: UDParsedBin(
			attrs=["ID", "ORTH", "LEMMA", "TAG", "POS", "HEAD", "DEP", "ENT_IOB", "ENT_TYPE"],
			store_user_data=True
		)


PARSING_CONFIG = ParsingConfig()
