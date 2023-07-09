from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DynamicExtractionsInfo:
	parsed_data: dict = field(default_factory=dict)
	mentions_by_event: Dict[int, List[dict]] = field(default_factory=dict)

	# sorted events for each sentence
	sorted_noun_events: List[List[int]] = field(default_factory=list)
	sorted_verb_events: List[List[int]] = field(default_factory=list)

	sentence_id: int = field(default=0)
	sent_shift_idx: int = field(default=0)
