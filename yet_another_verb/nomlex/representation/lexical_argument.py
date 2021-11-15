from typing import List

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

from yet_another_verb.nomlex.constants import ArgumentType


@dataclass_json
@dataclass
class LexicalArgument:
	arg_type: ArgumentType
	plural: bool = field(default=False)
	subjunct: bool = field(default=False)
	attributes: List[str] = field(default_factory=list)
	controlled: List[ArgumentType] = field(default_factory=list)
