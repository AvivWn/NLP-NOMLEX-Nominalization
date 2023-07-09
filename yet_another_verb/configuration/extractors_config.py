from enum import Enum
from os.path import join

from yet_another_verb.arguments_extractor.extraction import ArgumentType
from yet_another_verb.arguments_extractor.extractors.verb_references_based.method_params import MethodParams
from yet_another_verb.arguments_extractor.extractors.verb_references_based.verb_references_args_extractor import \
	DependencyArgAVGArgsExtractor, DependencyArgKNNArgsExtractor, DependencyExtKNNArgsExtractor
from yet_another_verb.configuration.encoding_config import EncodingLevel, EncodingFramework
from yet_another_verb.configuration.path_config import OUTER_PROJECT_PATH, NOMLEX_PATH, CONSTANTS_PATH
from yet_another_verb.dependency_parsing import DepRelation, NOUN_POSTAGS, POSTag, VERB_POSTAGS
from yet_another_verb.nomlex.nomlex_version import NomlexVersion
from yet_another_verb.arguments_extractor.extractors.constraints_based.nomlex_args_extractor import NomlexArgsExtractor
from yet_another_verb.arguments_extractor.extractors.constraints_based.patterns_args_extractors import VerbPatternsArgsExtractor
from yet_another_verb.nomlex.representation.constraints_map import ConstraintsMap


class ExtractorType(str, Enum):
	# Constraints based
	NOMLEX = "nomlex"
	VERB_PATTERNS = "verb-patterns"

	# Verb references based
	DEPENDENCY_AVG_ARG = "dependency-nearest-avg-argument"
	DEPENDENCY_ARG_KNN = "dependency-k-nearest-argument"
	DEPENDENCY_EXT_KNN = "dependency-k-nearest-extraction"


CONSTRAINTS_EXTRACTORS = [ExtractorType.NOMLEX, ExtractorType.VERB_PATTERNS]
VERB_REFERENCES_EXTRACTORS = [
	ExtractorType.DEPENDENCY_AVG_ARG, ExtractorType.DEPENDENCY_ARG_KNN, ExtractorType.DEPENDENCY_EXT_KNN]


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
		self.LEXICON_DIR = join(NOMLEX_PATH, "lexicons")
		self.LISP_LEXICON_DIR = join(self.LEXICON_DIR, "lisp")
		self.JSON_LEXICON_DIR = join(self.LEXICON_DIR, "json")
		self.PKL_LEXICON_DIR = join(self.LEXICON_DIR, "pkl")

		# References related configurations
		self.REFERENCES_ARG_TYPES = [ArgumentType.SUBJ, ArgumentType.OBJ, ArgumentType.PP]
		candidates_dependency_relations = [
			DepRelation.NSUBJ, DepRelation.NMOD_POSS, DepRelation.COMPOUND, DepRelation.NMOD]
		candidate_pp_values = [x.strip() for x in open(join(CONSTANTS_PATH, "nomlex_pp_values.txt")).readlines()]
		self.CANDIDATES_PP_VALUES = list(sorted(candidate_pp_values, reverse=True))  # specific first
		self.DEFAULT_METHOD_PARAMS = MethodParams()
		self.AVG_ARG_METHOD_PARAMS = MethodParams(
			dependency_relations=candidates_dependency_relations,
			redundant_threshold=0.56
		)
		self.ARG_KNN_METHOD_PARAMS = MethodParams(
			dependency_relations=candidates_dependency_relations,
			redundant_threshold=0.56
		)
		self.EXT_KNN_METHOD_PARAMS = MethodParams(
			dependency_relations=candidates_dependency_relations,
			redundant_threshold=0.6
		)
		self.METHOD_PARAMS_BY_EXTRACTOR = {
			ExtractorType.DEPENDENCY_AVG_ARG: self.AVG_ARG_METHOD_PARAMS,
			ExtractorType.DEPENDENCY_ARG_KNN: self.ARG_KNN_METHOD_PARAMS,
			ExtractorType.DEPENDENCY_EXT_KNN: self.EXT_KNN_METHOD_PARAMS
		}

		self.REFERENCES_PATH_BY_ENCODER = {
			(EncodingFramework.PRETRAINED_TORCH, "bert-large-uncased", EncodingLevel.HEAD_IDX_IN_ARG_CONTEXT):
				join(OUTER_PROJECT_PATH, "data/wiki40b/extracted/db-verbs/limited-verbs-by-patterns-bert-large"),
			(EncodingFramework.FASTTEXT, "fasttext-300-en", EncodingLevel.FULL_TEXT):
				"???"
		}


EXTRACTORS_CONFIG = ExtractorsConfig()


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


EXTRACTOR_BY_NAME = {
	ExtractorType.NOMLEX: NomlexArgsExtractor,
	ExtractorType.VERB_PATTERNS: VerbPatternsArgsExtractor,
	ExtractorType.DEPENDENCY_AVG_ARG: DependencyArgAVGArgsExtractor,
	ExtractorType.DEPENDENCY_ARG_KNN: DependencyArgKNNArgsExtractor,
	ExtractorType.DEPENDENCY_EXT_KNN: DependencyExtKNNArgsExtractor
}
NAME_BY_EXTRACTOR = {v: k for k, v in EXTRACTOR_BY_NAME.items()}
