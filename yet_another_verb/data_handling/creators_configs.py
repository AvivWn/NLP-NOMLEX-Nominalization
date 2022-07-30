from dataclasses import dataclass, field
from typing import Dict, Any

from yet_another_verb.arguments_extractor.extraction.argument.argument_type import NP_ARG_TYPES, PP_ARG_TYPES
from yet_another_verb.data_handling import WikiDatasetCreator, ParsedDatasetCreator, \
	ExtractedDatasetCreator, BIOArgsDatasetCreator, EncodedExtractionsCreator, ShuffledLinesDatasetCreator, \
	CombinedSQLitesDatasetCreator
from yet_another_verb.data_handling.file.file_extensions import TXT_EXTENSION, PARSED_EXTENSION, EXTRACTED_EXTENSION, \
	CSV_EXTENSION, DB_EXTENSION
from yet_another_verb.dependency_parsing import NOUN_POSTAGS, POSTag
from yet_another_verb.factories.dependency_parser_factory import DependencyParserFactory
from yet_another_verb.factories.extractor_factory import ExtractorFactory
from yet_another_verb.factories.verb_translator_factory import VerbTranslatorFactory


@dataclass
class DatasetConfig:
	creator_type: type
	subdir_name: str = field(default="")
	args: Dict[str, Any] = field(default_factory=dict)


EXTENSION_BY_CREATOR_TYPE = {
	WikiDatasetCreator: TXT_EXTENSION,
	ShuffledLinesDatasetCreator: TXT_EXTENSION,
	ParsedDatasetCreator: PARSED_EXTENSION,
	ExtractedDatasetCreator: EXTRACTED_EXTENSION,
	BIOArgsDatasetCreator: CSV_EXTENSION,
	EncodedExtractionsCreator: DB_EXTENSION,
	CombinedSQLitesDatasetCreator: DB_EXTENSION
}

DIR_BY_CREATORE_TYPE = {
	ParsedDatasetCreator: "parsed",
	ExtractedDatasetCreator: "extracted",
	BIOArgsDatasetCreator: "bio-args",
	EncodedExtractionsCreator: "encoded-extractions",
	CombinedSQLitesDatasetCreator: "encoded-extractions"
}

DATASET_CREATORS_BY_TYPE = {
	ParsedDatasetCreator: lambda kwargs: ParsedDatasetCreator(
		**kwargs,
		dependency_parser=DependencyParserFactory(**kwargs)()),
	ExtractedDatasetCreator: lambda kwargs: ExtractedDatasetCreator(
		**kwargs,
		args_extractor=ExtractorFactory(**kwargs)(),
		dependency_parser=DependencyParserFactory(**kwargs)()),
	BIOArgsDatasetCreator: lambda kwargs: BIOArgsDatasetCreator(
		**kwargs,
		verb_translator=VerbTranslatorFactory(**kwargs)()),
	EncodedExtractionsCreator: lambda kwargs: EncodedExtractionsCreator(
		**kwargs,
		dependency_parser=DependencyParserFactory(**kwargs)(),
		args_extractor=ExtractorFactory(**kwargs)(),
		verb_translator=VerbTranslatorFactory(**kwargs)(),
	)
}

LIMITED_WORDS = [
	"participate", "respond", "define", "require", "evaluate", "invest", "separate", "construct", "intercede", "affect",
	"describe", "restrict", "release", "classify", "acquire", "analyze", "analyse", "maintain", "assess", "control",
	"deploy", "visit", "possess", "develop", "pollute", "resist", "expand", "aspire", "adapt", "accept", "publish",
	"influence", "explain", "continue", "access", "interpret", "support", "regulate", "convert", "nominate",
	"contribute", "approve", "transport", "assist", "apply", "understand", "strike", "injure", "proclaim", "migrate",
	"predict", "use", "legislate", "reform", "defend", "perform", "attack", "violate", "absorb", "study", "illuminate",
	"settle", "depict", "vary", "introduce", "operate", "direct", "abuse", "confirm", "produce", "dominate", "rule",
	"plan", "aid", "discriminate", "treat", "advance", "examine", "harm", "behave", "trade", "expose", "diverge",
	"transform", "prepare", "involve", "resolve", "move", "provide", "accompany", "imitate", "navigate",
	'distribute', 'blockade', 'attend', 'protest', 'cover', 'struggle', 'broadcast', 'investigate', 'intervene',
	'shrink', 'interview', 'achieve', 'travel', 'erode', 'educate', 'satisfy', 'link', 'distinguish', 'declare',
	'censor', 'conclude', 'sell', 'commit', 'excavate', 'report', 'improve', 'allocate', 'own', 'approach', 'appear',
	'limit', 'tour', 'consume', 'compose', 'relocate', 'enroll', 'discharge', 'collaborate', 'imprison', 'deliver',
	'circulate', 'admit', 'damage', 'prefer'
]

UD_PARSED_CONFIGS = [DatasetConfig(ParsedDatasetCreator, "en_ud_model_lg-2.0.0")]
NOMLEX_EXTRACTED_CONFIG = UD_PARSED_CONFIGS + [DatasetConfig(ExtractedDatasetCreator, "nomlex")]

DATASET_CONFIGS_BY_TYPE = {
	"wiki40b": [DatasetConfig(WikiDatasetCreator)],
	"shuffled": [DatasetConfig(ShuffledLinesDatasetCreator)],
	"parsed/en_ud_model_lg-2.0.0": UD_PARSED_CONFIGS,
	"extracted/nomlex": NOMLEX_EXTRACTED_CONFIG,

	"bio-args/noun-args-by-v1-skip-predicate": NOMLEX_EXTRACTED_CONFIG + [
		DatasetConfig(
			BIOArgsDatasetCreator,
			"noun-args-by-v1-skip-predicate",
			{
				"limited_types": NP_ARG_TYPES + PP_ARG_TYPES,
				"limited_postags": NOUN_POSTAGS,
				"use_base_verb": True,
				"avoid_outside_tag": False,
				"tag_predicate": False
			})],

	"encoded-extractions/all": UD_PARSED_CONFIGS + [
		DatasetConfig(
			EncodedExtractionsCreator,
			"all",
			{
				"model_name": "bert-base-uncased",
				"device": "cuda"
			}
		)],

	"encoded-extractions/limited-verbs": UD_PARSED_CONFIGS + [
		DatasetConfig(
			EncodedExtractionsCreator,
			"limited-verbs",
			{
				"limited_postags": [POSTag.VERB],
				"limited_words": LIMITED_WORDS,
				"model_name": "bert-base-uncased",
				"device": "cuda"
			})],

	"combined-sqlite": [DatasetConfig(CombinedSQLitesDatasetCreator)],
}
