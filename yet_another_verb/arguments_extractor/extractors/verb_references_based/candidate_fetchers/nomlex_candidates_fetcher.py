from yet_another_verb import NomlexArgsExtractor
from yet_another_verb.arguments_extractor.extraction.utils.argument_utils import get_argument_head_idx, \
	specify_pp_type_in_arg
from yet_another_verb.arguments_extractor.extraction.utils.combination import combine_extractions
from yet_another_verb.arguments_extractor.extraction import ExtractedArguments
from yet_another_verb.arguments_extractor.extraction.utils.reconstruction import reconstruct_extraction
from yet_another_verb.arguments_extractor.extractors.verb_references_based.candidate_fetchers import \
	CandidatesFetcher, DependencyCandidatesFetcher
from yet_another_verb.arguments_extractor.extractors.verb_references_based.method_params import MethodParams
from yet_another_verb.configuration.extractors_config import EXTRACTORS_CONFIG
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText


class NomlexCandidatesFetcher(CandidatesFetcher):
	def __init__(self, nomlex_version=EXTRACTORS_CONFIG.NOMLEX_VERSION, use_cache=True):
		self.dependency_candidate_fetcher = DependencyCandidatesFetcher()
		self.nomlex_args_extractor = NomlexArgsExtractor(nomlex_version)
		self.use_cache = use_cache
		self._chached_by_id = {}

	def extract_by_nomlex_with_cache(self, words, predicate_idx):
		text = " ".join([str(word) for word in words])
		cache_id = (text, predicate_idx)

		if all(isinstance(word, str) for word in words):
			parsed_text = text
		else:
			parsed_text = words

		if self.use_cache and cache_id in self._chached_by_id:
			extractions = self._chached_by_id[cache_id]
		else:
			multiword_ext = self.nomlex_args_extractor.extract_multiword(parsed_text, limited_indices=[predicate_idx])
			extractions = multiword_ext.extractions_per_idx.get(predicate_idx)

		self._chached_by_id[cache_id] = extractions

		if extractions is None:
			return extractions

		return [reconstruct_extraction(ext) for ext in extractions]

	def fetch_arguments(self, word_idx: int, words: ParsedText, method_params: MethodParams) -> ExtractedArguments:
		nomlex_extractions = self.extract_by_nomlex_with_cache(words, word_idx)
		if nomlex_extractions is None:
			return []

		args = combine_extractions(nomlex_extractions, safe_combine=method_params.consider_only_determined_args).args
		args = [arg for arg in args if word_idx not in arg.arg_indices and arg.arg_type in method_params.arg_types]
		arg_by_head_idx = {get_argument_head_idx(words, arg): arg for arg in args}

		# The arguments of noun should include every AMOD relation (for potential adjective)
		# Should it be a part of the nomlex adaptation?
		if not method_params.consider_only_determined_args and method_params.consider_adj_relations:
			extra_args = self.dependency_candidate_fetcher.fetch_arguments(word_idx, words, method_params)
			for arg in extra_args:
				head_idx = get_argument_head_idx(words, arg)
				if head_idx not in arg_by_head_idx:
					arg_by_head_idx[head_idx] = arg

		if method_params.consider_pp_type:
			for arg in arg_by_head_idx.values():
				specify_pp_type_in_arg(words, arg)

		return list(arg_by_head_idx.values())
