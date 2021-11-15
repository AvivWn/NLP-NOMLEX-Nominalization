from typing import Dict

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

from yet_another_verb.nomlex.constants import SubcatType, ArgumentType
from yet_another_verb.nomlex.representation.lexical_argument import LexicalArgument
from yet_another_verb.nomlex.representation.constraints_map import ORConstraintsMaps


@dataclass_json
@dataclass
class LexicalSubcat:
	subcat_type: SubcatType
	constraints_maps: ORConstraintsMaps = field(default_factory=list)
	lexical_args: Dict[ArgumentType, LexicalArgument] = field(default_factory=dict)
