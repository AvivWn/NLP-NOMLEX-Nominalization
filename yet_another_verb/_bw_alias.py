import sys

from yet_another_verb.arguments_extractor.extraction.argument import extracted_argument, argument_type
from yet_another_verb.data_handling.bytes.compressed import compressed_parsed_text
from yet_another_verb.dependency_parsing import dependency_relation, postag

sys.modules["yet_another_verb.arguments_extractor.extraction.extracted_argument"] = extracted_argument
sys.modules["yet_another_verb.nomlex.constants.argument_type"] = argument_type
sys.modules["yet_another_verb.nomlex.constants.word_relation"] = dependency_relation
sys.modules["yet_another_verb.nomlex.constants.word_postag"] = postag
sys.modules["yet_another_verb.dependency_parsing.compressed_parsed_text"] = compressed_parsed_text
