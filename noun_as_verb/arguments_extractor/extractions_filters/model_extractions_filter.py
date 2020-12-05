from .extractions_filter import ExtractionsFilter
from noun_as_verb.model_based.types_predictor import TypesPredictor
from noun_as_verb.utils import aggregate_to_dict


class RulesExtractionsFilter(ExtractionsFilter):
	model: TypesPredictor

	def __init__(self, model: TypesPredictor):
		super().__init__()
		self.model = model

	def _choose_by_model(self, extractions, predicate, original_extractions):
		# Determine the complement type of uncertain candidates, with a model
		candidates_args = aggregate_to_dict([e.get_candidates_types() for e in extractions])
		candidates_args = self.model.determine_args_type(candidates_args, predicate)

		# Clean extractions from arguments that weren't chosen
		extractions = [e for e in extractions if e.get_filtered(candidates_args).isin(original_extractions)]
		return extractions

	def filter(self, extractions, predicate):
		original_extractions = extractions
		extractions = self._choose_by_model(extractions, predicate, original_extractions)
		extractions = super().filter(extractions, predicate)
		return extractions
