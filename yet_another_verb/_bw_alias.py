import sys
from yet_another_verb.arguments_extractor.extraction.argument import extracted_argument, argument_type
sys.modules["yet_another_verb.arguments_extractor.extraction.extracted_argument"] = extracted_argument
sys.modules["yet_another_verb.nomlex.constants.argument_type"] = argument_type
