from itertools import chain

from yet_another_verb.data_handling import WikiDatasetCreator, ParsedDatasetCreator, \
	ExtractedFromParsedDatasetCreator, BIOArgsDatasetCreator, EncodedExtractionsCreator, ShuffledLinesDatasetCreator, \
	CombinedSQLitesDatasetCreator, EncodedExtractionsExpander, ExtractedFromDBDatasetCreator, TXTFileHandler
from yet_another_verb.factories.dependency_parser_factory import DependencyParserFactory
from yet_another_verb.factories.extractor_factory import ExtractorFactory
from yet_another_verb.factories.encoder_factory import EncoderFactory
from yet_another_verb.factories.verb_translator_factory import VerbTranslatorFactory


LIMITED_VERBS_BY_TYPE = {
	"paraphrasing-verbs": TXTFileHandler(handle_multi_line=True).load("constants/paraphrasing_verbs.txt"),
	"nomlex-verbs": TXTFileHandler(handle_multi_line=True).load("constants/nomlex_verbs_750.txt"),
}
LIMITED_VERBS_BY_TYPE["all-limited-verbs"] = list(chain(*LIMITED_VERBS_BY_TYPE.values()))


DATASET_CREATOR_BY_TYPE = {
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
		encoder=EncoderFactory(**kwargs)(),
		limited_verbs=LIMITED_VERBS_BY_TYPE[kwargs["limited_verbs_type"]]
	),
	EncodedExtractionsExpander: lambda kwargs: EncodedExtractionsExpander(
		**kwargs,
		dependency_parser=DependencyParserFactory(**kwargs)(),
		args_extractor=ExtractorFactory(**kwargs)(),
		verb_translator=VerbTranslatorFactory(**kwargs)(),
		encoder=EncoderFactory(**kwargs)()
	)
}


DATASET_CREATOR_TYPE_BY_NAME = {
	"wiki40b": WikiDatasetCreator,
	"shuffled": ShuffledLinesDatasetCreator,
	"parsed": ParsedDatasetCreator,
	"extracted": ExtractedFromParsedDatasetCreator,
	"bio-args": BIOArgsDatasetCreator,
	"encoded-extractions": EncodedExtractionsCreator,
	"encoded-extractions-expander": EncodedExtractionsExpander,
	"combined-sqlite": CombinedSQLitesDatasetCreator,
	"extracted-from-db": ExtractedFromDBDatasetCreator
}
