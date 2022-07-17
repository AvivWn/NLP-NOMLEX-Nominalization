from typing import Dict, List

from yet_another_verb.nomlex.constants import LexiconType, SubcatType, ArgumentValue
from yet_another_verb.arguments_extractor.extraction import ArgumentType

NotConstraint = Dict[ArgumentType, Dict[ArgumentValue, List[str]]]

SUBCAT_NOT_CONSTRAINTS = {

}

EVERY_SUBCAT_NOT_CONSTRAINTS = {
	LexiconType.VERB: [
		{
			ArgumentType.SUBJ: {ArgumentValue.NSUBJ: []},
			ArgumentType.OBJ: {ArgumentValue.NSUBJPASS: []}
		},
		{
			ArgumentType.SUBJ: {ArgumentValue.NP: ["by"]},
			ArgumentType.OBJ: {ArgumentValue.DOBJ: []}
		}
	]
}


def get_extra_not_constraints(subcat_type: SubcatType, lexicon_type: LexiconType) -> List[NotConstraint]:
	return SUBCAT_NOT_CONSTRAINTS.get(subcat_type, {}).get(lexicon_type, []) + \
		EVERY_SUBCAT_NOT_CONSTRAINTS.get(lexicon_type, [])
