from typing import Dict

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

from lexical_entry import LexicalEntry


@dataclass_json
@dataclass
class Lexicon:
	entries: Dict[str, LexicalEntry] = field(default_factory=dict)
