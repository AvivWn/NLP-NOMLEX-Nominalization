from ..extraction import Extraction
from .extraction_representation import ExtractionRepresentation


class SpanRepresentation(ExtractionRepresentation):
	def __init__(self):
		super().__init__()

	def represent(self, extraction: Extraction, trim_arguments=True):
		extraction_dict = {}

		# Cleans the extraction, deletes duplicates between args and translates args into spans
		for arg in extraction.get_match().values():
			arg_span = arg.as_span(trim_arguments)

			if arg_span:
				extraction_dict[arg.get_name()] = arg_span

		return extraction_dict
