from os.path import join, dirname

from yet_another_verb import nomlex
from yet_another_verb.arguments_extractor.extraction import ArgumentType
from yet_another_verb.dependency_parsing import DepRelation, NOUN_POSTAGS, POSTag, VERB_POSTAGS
from yet_another_verb.nomlex.nomlex_version import NomlexVersion
from yet_another_verb.arguments_extractor.extractors.nomlex_args_extractor import NomlexArgsExtractor
from yet_another_verb.arguments_extractor.extractors.patterns_args_extractors import VerbPatternsArgsExtractor
from yet_another_verb.nomlex.representation.constraints_map import ConstraintsMap


class ExtractorsConfig:
	def __init__(
			self,
			extractor="nomlex",
			version=NomlexVersion.V2,
			use_nomlex_cache=True
	):
		self.EXTRACTOR = extractor

		# nomlex related configurations
		self.USE_NOMLEX_CACHE = use_nomlex_cache
		self.NOMLEX_VERSION = version
		self.LEXICON_DIR = join(dirname(nomlex.__file__), "lexicons")
		self.LISP_LEXICON_DIR = join(self.LEXICON_DIR, "lisp")
		self.JSON_LEXICON_DIR = join(self.LEXICON_DIR, "json")
		self.PKL_LEXICON_DIR = join(self.LEXICON_DIR, "pkl")


EXTRACTORS_CONFIG = ExtractorsConfig()

EXTRACTOR_BY_NAME = {
	"nomlex": NomlexArgsExtractor,
	"verb-patterns": VerbPatternsArgsExtractor
}
NAME_BY_EXTRACTOR = {v: k for k, v in EXTRACTOR_BY_NAME.items()}


PP_PATTERN = ConstraintsMap(
	arg_type=ArgumentType.PP, dep_relations=[DepRelation.NMOD], postags=NOUN_POSTAGS, required=False,
	relatives_constraints=[
		ConstraintsMap(arg_type=ArgumentType.PP, dep_relations=[DepRelation.CASE], postags=[POSTag.IN], required=True)]
)
ACTIVE_VERB_PATTERN = ConstraintsMap(
	postags=VERB_POSTAGS, required=True,
	relatives_constraints=[
		ConstraintsMap(arg_type=ArgumentType.SUBJ, dep_relations=[DepRelation.NSUBJ], postags=NOUN_POSTAGS, required=True),
		ConstraintsMap(arg_type=ArgumentType.OBJ, dep_relations=[DepRelation.DOBJ], postags=NOUN_POSTAGS, required=False),
		PP_PATTERN
	])
PASSIVE_VERB_PATTERN = ConstraintsMap(
	postags=VERB_POSTAGS, required=True,
	relatives_constraints=[
		ConstraintsMap(
			arg_type=ArgumentType.OBJ, dep_relations=[DepRelation.NSUBJPASS], postags=NOUN_POSTAGS, required=True),
		ConstraintsMap(
			arg_type=ArgumentType.SUBJ, dep_relations=[DepRelation.NMOD], postags=NOUN_POSTAGS, required=False,
			relatives_constraints=[
				ConstraintsMap(dep_relations=[DepRelation.CASE], postags=[POSTag.IN], values=["by"], required=True)]
		),
		ConstraintsMap(dep_relations=[DepRelation.AUXPASS], required=True),
		PP_PATTERN
	])
VERB_PATTERNS = [ACTIVE_VERB_PATTERN, PASSIVE_VERB_PATTERN]
