from yet_another_verb.dependency_parsing.spacy.spacy_parser import SpacyParser

SPACY_ENGINE = "spacy"


parser_by_engine = {
	SPACY_ENGINE: SpacyParser
}
engine_by_parser = {v: k for k, v in parser_by_engine.items()}
