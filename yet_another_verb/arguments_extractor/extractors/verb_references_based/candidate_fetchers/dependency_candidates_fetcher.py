from yet_another_verb.arguments_extractor.extraction import ExtractedArgument, ExtractedArguments, ArgumentType
from yet_another_verb.arguments_extractor.extraction.utils.indices import get_indices_as_range
from yet_another_verb.arguments_extractor.extractors.verb_references_based.candidate_fetchers.candidates_fetcher import \
	CandidatesFetcher
from yet_another_verb.arguments_extractor.extractors.verb_references_based.method_params import MethodParams
from yet_another_verb.dependency_parsing import DepRelation
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText


class DependencyCandidatesFetcher(CandidatesFetcher):
	def fetch_arguments(self, word_idx: int, parsed_text: ParsedText, method_params: MethodParams) -> ExtractedArguments:
		predicate = parsed_text[word_idx]

		dependency_relations = method_params.dependency_relations
		if method_params.consider_adj_relations:
			dependency_relations = set(method_params.dependency_relations + [DepRelation.AMOD])

		return [
			ExtractedArgument(*get_indices_as_range(c.subtree_indices), head_idx=c.i, arg_type=ArgumentType.REDUNDANT)
			for c in predicate.children if c.dep in dependency_relations]
