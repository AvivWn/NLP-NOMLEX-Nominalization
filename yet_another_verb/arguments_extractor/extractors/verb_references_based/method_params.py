from typing import Optional, List

from attr import dataclass
from dataclasses import field

from yet_another_verb.arguments_extractor.extraction.argument.argument_type import ArgumentTypes
from yet_another_verb.arguments_extractor.extractors.verb_references_based.similarity_scorer.cosine_scorer import \
	CosineScorer
from yet_another_verb.arguments_extractor.extractors.verb_references_based.similarity_scorer.similarity_scorer import \
	SimilarityScorer
from yet_another_verb.dependency_parsing import DepRelation


@dataclass
class MethodParams:
	references_amount: Optional[int] = None
	arg_types: Optional[ArgumentTypes] = None  # Useful for fetching and labeling

	# Candidate fetching related
	dependency_relations: List[DepRelation] = field(default_factory=list)
	consider_adj_relations: bool = True
	consider_only_determined_args: bool = False  # relevant for NomLex

	# Argument labeling related
	already_normalized = True
	similarity_scorer: SimilarityScorer = CosineScorer(normalized_left=already_normalized)
	max_repeated_args: Optional[int] = 1
	consider_pp_type: bool = True
	redundant_threshold: float = None
	k_neighbors: int = 5
