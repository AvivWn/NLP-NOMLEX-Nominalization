from dataclasses import dataclass, field
from typing import Dict, Any

from yet_another_verb.arguments_extractor.extraction.argument.argument_type import NP_ARG_TYPES, PP_ARG_TYPES
from yet_another_verb.data_handling import WikiDatasetCreator, ParsedDatasetCreator, \
	ExtractedFromParsedDatasetCreator, BIOArgsDatasetCreator, EncodedExtractionsCreator, ShuffledLinesDatasetCreator, \
	CombinedSQLitesDatasetCreator, EncodedExtractionsExpander, ExtractedFromDBDatasetCreator
from yet_another_verb.dependency_parsing import NOUN_POSTAGS, POSTag
from yet_another_verb.factories.dependency_parser_factory import DependencyParserFactory
from yet_another_verb.factories.extractor_factory import ExtractorFactory
from yet_another_verb.factories.encoder_factory import EncoderFactory
from yet_another_verb.factories.verb_translator_factory import VerbTranslatorFactory
from yet_another_verb.sentence_encoding.argument_encoding.encoding_level import EncodingLevel


@dataclass
class DatasetConfig:
	creator_type: type
	args: Dict[str, Any] = field(default_factory=dict)


DATASET_CREATORS_BY_TYPE = {
	ParsedDatasetCreator: lambda kwargs: ParsedDatasetCreator(
		**kwargs,
		dependency_parser=DependencyParserFactory(**kwargs)()),
	ExtractedFromParsedDatasetCreator: lambda kwargs: ExtractedFromParsedDatasetCreator(
		**kwargs,
		args_extractor=ExtractorFactory(**kwargs)()),
	BIOArgsDatasetCreator: lambda kwargs: BIOArgsDatasetCreator(
		**kwargs,
		verb_translator=VerbTranslatorFactory(**kwargs)()),
	EncodedExtractionsCreator: lambda kwargs: EncodedExtractionsCreator(
		**kwargs,
		args_extractor=ExtractorFactory(**kwargs)(),
		verb_translator=VerbTranslatorFactory(**kwargs)(),
		encoder=EncoderFactory(**kwargs)()
	),
	EncodedExtractionsExpander: lambda kwargs: EncodedExtractionsExpander(
		**kwargs,
		dependency_parser=DependencyParserFactory(**kwargs)(),
		args_extractor=ExtractorFactory(**kwargs)(),
		verb_translator=VerbTranslatorFactory(**kwargs)(),
		encoder=EncoderFactory(**kwargs)()
	)
}

LIMITED_VERBS = [
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

APPEND_DATASET_TYPE_TO_PATH_CREATORS = [ParsedDatasetCreator, BIOArgsDatasetCreator, EncodedExtractionsCreator]


DATASET_CONFIGS_BY_TYPE = {
	"wiki40b": DatasetConfig(WikiDatasetCreator),
	"shuffled": DatasetConfig(ShuffledLinesDatasetCreator),
	"parsed/en_ud_model_lg-2.0.0": DatasetConfig(
		ParsedDatasetCreator, {"parsing_engine": "spacy", "parser_name": "en_ud_model_lg"}),
	"extracted/nomlex": DatasetConfig(ExtractedFromParsedDatasetCreator, {"extraction_mode": "nomlex"}),

	"bio-args/noun-args-by-v1-skip-predicate":
		DatasetConfig(
			BIOArgsDatasetCreator,
			{
				"limited_types": NP_ARG_TYPES + PP_ARG_TYPES,
				"limited_postags": NOUN_POSTAGS,
				"use_base_verb": True,
				"avoid_outside_tag": False,
				"tag_predicate": False
			}),

	"encoded-extractions/all": DatasetConfig(EncodedExtractionsCreator),

	"encoded-extractions/limited-verbs":
		DatasetConfig(
			EncodedExtractionsCreator,
			{
				"limited_postags": [POSTag.VERB, POSTag.NOUN],
				"limited_verbs": LIMITED_VERBS,
			}),

	"encoded-extractions/expand": DatasetConfig(EncodedExtractionsExpander),

	"combined-sqlite": DatasetConfig(CombinedSQLitesDatasetCreator),

	"extracted-from-db/noun":
		DatasetConfig(
			ExtractedFromDBDatasetCreator,
			{
				"parsing_engine": "spacy",
				"parser_name": "en_ud_model_lg",
				"extractor": "nomlex",
				"encoding_framework": "pretrained_torch",
				"encoder_name": "bert-base-uncased",
				"encoding_level": EncodingLevel.HEAD_IDX_IN_SENTENCE_CONTEXT,
				"limited_postags": [POSTag.NOUN]
			}),

	"extracted-from-db/verb":
		DatasetConfig(
			ExtractedFromDBDatasetCreator,
			{
				"parsing_engine": "spacy",
				"parser_name": "en_ud_model_lg",
				"extractor": "nomlex",
				"encoding_framework": "pretrained_torch",
				"encoder_name": "bert-base-uncased",
				"encoding_level": EncodingLevel.HEAD_IDX_IN_SENTENCE_CONTEXT,
				"limited_postags": [POSTag.VERB]
			}),
}
