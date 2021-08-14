from typing import List
from dataclasses import dataclass

from yet_another_verb.nomlex.constants import ArgumentType


@dataclass
class ExtractedArgument:
	arg_idxs: List[int]
	arg_type: ArgumentType
