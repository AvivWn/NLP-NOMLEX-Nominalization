from typing import List, Dict, Optional
from copy import deepcopy
from itertools import chain
from collections import defaultdict

from yet_another_verb.nomlex.constants import LexiconType, ArgumentValue, ArgumentType, \
	WordRelation, POSTag, SubcatType
from yet_another_verb.nomlex.constants.word_postag import NOUN_POSTAGS, VERB_POSTAGS, \
	ADVERB_POSTAGS, ADJECTIVE_POSTAGS
from yet_another_verb.nomlex.representation.constraints_map import ConstraintsMap, ORConstraintsMaps, ANDConstraintsMaps

TO_PREP = "to"
AS_PREP = "as"


def _exapnd_constraint_separately(constraint_map: ConstraintsMap, relatives_constraints: ORConstraintsMaps) -> ORConstraintsMaps:
	if len(relatives_constraints) == 0:
		return [constraint_map]

	resulted_maps = []
	for relative_constraints in relatives_constraints:
		expanded_consrtaint = deepcopy(constraint_map)
		expanded_consrtaint.relatives_constraints.append(relative_constraints)
		resulted_maps.append(expanded_consrtaint)

	return resulted_maps


def _expand_constraints_separately(constraint_maps: ORConstraintsMaps, relatives_constraints: ORConstraintsMaps) -> ORConstraintsMaps:
	resulted_maps = []
	for constraint_map in constraint_maps:
		resulted_maps += _exapnd_constraint_separately(constraint_map, relatives_constraints)

	return resulted_maps


def _get_mark_map(values: List[str], arg_type: ArgumentType, postags: List[POSTag] = None) -> ConstraintsMap:
	postags = [] if postags is None else postags
	return ConstraintsMap(word_relations=[WordRelation.MARK], values=values, postags=postags, arg_type=arg_type)


def _get_advmod_map(values: List[str], arg_type: ArgumentType, postags: List[POSTag] = None) -> ConstraintsMap:
	postags = [] if postags is None else postags
	return ConstraintsMap(word_relations=[WordRelation.ADVMOD], values=values, postags=postags, arg_type=arg_type)


def _get_np_maps(word_relations: List[WordRelation], arg_type: ArgumentType) -> ORConstraintsMaps:
	return [ConstraintsMap(
		word_relations=word_relations, postags=NOUN_POSTAGS + [POSTag.PRP_POSS], arg_type=arg_type,
		relatives_constraints=[ConstraintsMap(values=["'s"], required=False)]
	)]


def _get_possessive_maps(arg_type: ArgumentType) -> ORConstraintsMaps:
	possessive_relations = [WordRelation.NSUBJ, WordRelation.NMOD_POSS]

	return [
		ConstraintsMap(
			word_relations=possessive_relations,
			postags=[POSTag.PRP_POSS],
			arg_type=arg_type),
		ConstraintsMap(
			word_relations=possessive_relations,
			postags=NOUN_POSTAGS,
			relatives_constraints=[ConstraintsMap(values=["'s"])],
			arg_type=arg_type)
	]


def _get_adjective_maps(word_relations: List[WordRelation], arg_type: ArgumentType) -> ORConstraintsMaps:
	return [ConstraintsMap(word_relations=word_relations, postags=ADJECTIVE_POSTAGS, arg_type=arg_type)]


def _get_adverb_maps(arg_type: ArgumentType):
	return [ConstraintsMap(word_relations=[WordRelation.ADVMOD], postags=ADVERB_POSTAGS, arg_type=arg_type)]


def _get_to_inf_maps(word_relations: List[WordRelation], arg_type: ArgumentType) -> ORConstraintsMaps:
	return [ConstraintsMap(
		word_relations=word_relations,
		postags=[POSTag.VB],
		relatives_constraints=[_get_mark_map(["to"], arg_type, [POSTag.TO])],
		arg_type=arg_type
	), ConstraintsMap(
		# word_relations=word_relations,
		relatives_constraints=[
			_get_mark_map(["to"], arg_type, [POSTag.TO]),
			ConstraintsMap(
				word_relations=[WordRelation.COP],
				postags=[POSTag.VB],
				values=["be"],
				arg_type=arg_type
			)
		],
		arg_type=arg_type
	)]


def _get_ing_maps(
		word_relations: List[WordRelation],
		arg_type: ArgumentType,
		possessive: bool = False) -> ORConstraintsMaps:
	possessive_constraints = [] if not possessive else _get_possessive_maps(arg_type)

	maps = [
		ConstraintsMap(
			word_relations=word_relations,
			postags=[POSTag.VBG],
			arg_type=arg_type
		),
		ConstraintsMap(
			word_relations=word_relations,
			relatives_constraints=[
				ConstraintsMap(
					word_relations=[WordRelation.COP],
					postags=[POSTag.VBG],
					values=["being"],
					arg_type=arg_type
				)
			],
			arg_type=arg_type
		)
	]

	return _expand_constraints_separately(maps, possessive_constraints)


def _get_sbar_maps(
		word_relations: List[WordRelation],
		arg_type: ArgumentType,
		relatives_constraints: ANDConstraintsMaps = None,
		tensed_only: bool = False,
		untensed_only: bool = False
) -> ORConstraintsMaps:
	assert tensed_only ^ untensed_only or (not tensed_only and not untensed_only)

	relatives_constraints = [] if relatives_constraints is None else relatives_constraints

	if tensed_only:
		relatives_constraints += [ConstraintsMap(
			word_relations=[WordRelation.NSUBJ, WordRelation.NSUBJPASS],
			postags=NOUN_POSTAGS,
			arg_type=arg_type
		)]

	if untensed_only:
		relatives_constraints += [_get_mark_map(["to"], arg_type, [POSTag.TO])]

	return [ConstraintsMap(
		word_relations=word_relations,
		postags=[POSTag.VB] if untensed_only else VERB_POSTAGS,
		relatives_constraints=relatives_constraints,
		arg_type=arg_type
	), ConstraintsMap(
		word_relations=word_relations,
		relatives_constraints=relatives_constraints + [ConstraintsMap(
			word_relations=[WordRelation.COP],
			postags=VERB_POSTAGS,
			values=["is", "are", "am", "was", "were", "be"],
			arg_type=arg_type
		)],
		arg_type=arg_type
	)]


def _get_that_s_maps(word_relations: List[WordRelation], arg_type: ArgumentType) -> ORConstraintsMaps:
	return _get_sbar_maps(word_relations, arg_type, [_get_mark_map(["that"], arg_type)], tensed_only=True)


def _get_whether_s_maps(word_relations: List[WordRelation], arg_type: ArgumentType) -> ORConstraintsMaps:
	return _get_sbar_maps(word_relations, arg_type, [_get_mark_map(["whether"], arg_type)])


def _get_what_s_maps(word_relations: List[WordRelation], arg_type: ArgumentType) -> ORConstraintsMaps:
	return _get_sbar_maps(word_relations, arg_type, [ConstraintsMap(
		values=["what"],
		word_relations=[WordRelation.NSUBJ, WordRelation.DOBJ],
		arg_type=arg_type
	)])


def _get_if_s_maps(word_relations: List[WordRelation], arg_type: ArgumentType) -> ORConstraintsMaps:
	return _get_sbar_maps(word_relations, arg_type, [_get_mark_map(["if"], arg_type)])


def _get_where_s_maps(word_relations: List[WordRelation], arg_type: ArgumentType) -> ORConstraintsMaps:
	return _get_sbar_maps(word_relations, arg_type, [_get_advmod_map(["where"], arg_type)])


def _get_when_s_maps(word_relations: List[WordRelation], arg_type: ArgumentType) -> ORConstraintsMaps:
	return _get_sbar_maps(word_relations, arg_type, [_get_advmod_map(["when"], arg_type)])


def _get_how_much_or_many_s_maps(word_relations: List[WordRelation], aux_value: str, arg_type: ArgumentType) -> ORConstraintsMaps:
	how_s_map = _get_advmod_map(["how"], arg_type)

	how_much_or_many_s_map = ConstraintsMap(
		word_relations=[WordRelation.ADVMOD, WordRelation.AMOD],
		values=[aux_value],
		relatives_constraints=[how_s_map],
		arg_type=arg_type
	)

	return _get_sbar_maps(
		word_relations,
		arg_type,
		[ConstraintsMap(
			word_relations=[WordRelation.DOBJ, WordRelation.NSUBJ, WordRelation.NSUBJPASS],
			relatives_constraints=[how_much_or_many_s_map],
			arg_type=arg_type
		)]
	) + _get_sbar_maps(
		word_relations,
		arg_type,
		[ConstraintsMap(
			word_relations=[WordRelation.DOBJ, WordRelation.NSUBJ, WordRelation.NSUBJPASS],
			relatives_constraints=[how_s_map],
			arg_type=arg_type,
			values=[aux_value]
		)]
	) + _get_sbar_maps(word_relations, arg_type, [how_much_or_many_s_map]) + \
		[ConstraintsMap(
				word_relations=[WordRelation.NMOD],
				postags=NOUN_POSTAGS,
				arg_type=arg_type,
				relatives_constraints=[how_much_or_many_s_map]
		)]


def _get_how_much_s_maps(word_relations: List[WordRelation], arg_type: ArgumentType) -> ORConstraintsMaps:
	return _get_how_much_or_many_s_maps(word_relations, "much", arg_type)


def _get_how_many_s_maps(word_relations: List[WordRelation], arg_type: ArgumentType) -> ORConstraintsMaps:
	return _get_how_much_or_many_s_maps(word_relations, "many", arg_type)


def _get_how_s_maps(word_relations: List[WordRelation], arg_type: ArgumentType, untensed_only: bool = False) -> ORConstraintsMaps:
	return _get_sbar_maps(word_relations, arg_type, [_get_advmod_map(["how"], arg_type)], untensed_only=untensed_only)


def _get_as_if_s_maps(word_relations: List[WordRelation], arg_type: ArgumentType) -> ORConstraintsMaps:
	if_mwe_map = ConstraintsMap(word_relations=[WordRelation.MWE], values=["if"], arg_type=arg_type)

	if_mark_map = ConstraintsMap(word_relations=[WordRelation.MARK], values=["if"], arg_type=arg_type)
	as_mark_map = ConstraintsMap(word_relations=[WordRelation.MARK], values=["as"], arg_type=arg_type)
	return _get_sbar_maps(
		word_relations,
		arg_type,
		[ConstraintsMap(
			word_relations=[WordRelation.MARK],
			values=["as"],
			relatives_constraints=[if_mwe_map],
			arg_type=arg_type
		)]
	) + _get_sbar_maps(
		word_relations,
		arg_type,
		[if_mark_map, as_mark_map]
	)


def _get_particle_maps(values: List[str], arg_type: ArgumentType) -> ORConstraintsMaps:
	return [ConstraintsMap(
		word_relations=[WordRelation.PRT, WordRelation.COMPOUND_PRT],
		values=values,
		postags=[POSTag.RP],
		arg_type=arg_type)
	]


ARG_CONSTRAINTS = {
	LexiconType.VERB: {
		ArgumentValue.NSUBJ: lambda arg_type: _get_np_maps([WordRelation.NSUBJ], arg_type),
		ArgumentValue.NSUBJPASS: lambda arg_type: _get_np_maps([WordRelation.NSUBJPASS], arg_type),
		ArgumentValue.DOBJ: lambda arg_type: _get_np_maps([WordRelation.DOBJ], arg_type),
		ArgumentValue.IOBJ: lambda arg_type: _get_np_maps([WordRelation.IOBJ], arg_type),

		ArgumentValue.ADMOD: lambda arg_type: _get_adverb_maps(arg_type),

		ArgumentValue.TO_INF: lambda arg_type: _get_to_inf_maps([WordRelation.ADVCL, WordRelation.XCOMP], arg_type),
		ArgumentValue.ING: lambda arg_type: _get_ing_maps([WordRelation.ADVCL, WordRelation.CCOMP, WordRelation.XCOMP], arg_type),
		ArgumentValue.POSSING: lambda arg_type: _get_ing_maps([WordRelation.CCOMP], arg_type, possessive=True),

		ArgumentValue.SBAR: lambda arg_type: _get_sbar_maps([WordRelation.CCOMP], arg_type, tensed_only=True),
		ArgumentValue.THAT_S: lambda arg_type: _get_that_s_maps([WordRelation.CCOMP], arg_type),
		ArgumentValue.IF_S: lambda arg_type: _get_if_s_maps([WordRelation.ADVCL], arg_type),
		ArgumentValue.WHAT_S: lambda arg_type: _get_what_s_maps([WordRelation.CCOMP], arg_type),
		ArgumentValue.WHETHER_S: lambda arg_type: _get_whether_s_maps([WordRelation.CCOMP], arg_type),
		ArgumentValue.WHERE_S: lambda arg_type: _get_where_s_maps([WordRelation.CCOMP, WordRelation.ADVCL], arg_type),
		ArgumentValue.WHEN_S: lambda arg_type: _get_when_s_maps([WordRelation.CCOMP, WordRelation.ADVCL], arg_type),
		ArgumentValue.HOW_MANY_S: lambda arg_type: _get_how_many_s_maps([WordRelation.CCOMP], arg_type),
		ArgumentValue.HOW_MUCH_S: lambda arg_type: _get_how_much_s_maps([WordRelation.CCOMP], arg_type),
		ArgumentValue.HOW_S: lambda arg_type: _get_how_s_maps([WordRelation.CCOMP], arg_type),
		ArgumentValue.HOW_TO_INF: lambda arg_type: _get_how_s_maps([WordRelation.XCOMP, WordRelation.CCOMP], arg_type, untensed_only=True),
		ArgumentValue.AS_IF_S: lambda arg_type: _get_as_if_s_maps([WordRelation.ADVCL], arg_type),
	},
	LexiconType.NOUN: {
		ArgumentValue.NSUBJ: lambda arg_type: _get_np_maps([WordRelation.NSUBJ], arg_type),
		ArgumentValue.N_N_MOD: lambda arg_type: _get_np_maps([WordRelation.COMPOUND], arg_type),
		ArgumentValue.DET_POSS: lambda arg_type: _get_possessive_maps(arg_type),

		ArgumentValue.ADMOD: lambda arg_type: _get_adverb_maps(arg_type),
		ArgumentValue.AJMOD: lambda arg_type: _get_adjective_maps([WordRelation.AMOD], arg_type),

		ArgumentValue.TO_INF: lambda arg_type: _get_to_inf_maps([WordRelation.ACL], arg_type),
		ArgumentValue.ING: lambda arg_type: _get_ing_maps([WordRelation.ACL, WordRelation.XCOMP], arg_type),
		ArgumentValue.POSSING: lambda arg_type: _get_ing_maps([WordRelation.ACL], arg_type, possessive=True),

		ArgumentValue.SBAR: lambda arg_type: _get_that_s_maps([WordRelation.CCOMP], arg_type),
		ArgumentValue.THAT_S: lambda arg_type: _get_that_s_maps([WordRelation.CCOMP], arg_type),
		ArgumentValue.AS_IF_S: lambda arg_type: _get_as_if_s_maps([WordRelation.ADVCL], arg_type),
	}
}


PREPOSITIONAL_ARG_CONSTRAINTS = {
	LexiconType.VERB: {
		ArgumentValue.NP: lambda arg_type: _get_np_maps([WordRelation.NMOD], arg_type),

		ArgumentValue.AJMOD: lambda arg_type: _get_adjective_maps([WordRelation.ADVCL], arg_type),

		ArgumentValue.TO_INF: lambda arg_type: _get_to_inf_maps([WordRelation.ADVCL], arg_type),
		ArgumentValue.ING: lambda arg_type: _get_ing_maps([WordRelation.ADVCL, WordRelation.XCOMP], arg_type),
		ArgumentValue.POSSING: lambda arg_type: _get_ing_maps([WordRelation.ADVCL], arg_type, possessive=True),

		ArgumentValue.IF_S: lambda arg_type: _get_if_s_maps([WordRelation.ADVCL], arg_type),
		ArgumentValue.WHAT_S: lambda arg_type: _get_what_s_maps([WordRelation.ADVCL], arg_type),
		ArgumentValue.WHETHER_S: lambda arg_type: _get_whether_s_maps([WordRelation.ADVCL], arg_type),
	},
	LexiconType.NOUN: {
		ArgumentValue.NP: lambda arg_type: _get_np_maps([WordRelation.NMOD], arg_type),

		ArgumentValue.AJMOD: lambda arg_type: _get_adjective_maps([WordRelation.ACL], arg_type),

		ArgumentValue.TO_INF: lambda arg_type: _get_to_inf_maps([WordRelation.ACL], arg_type),
		ArgumentValue.ING: lambda arg_type: _get_ing_maps([WordRelation.ACL, WordRelation.ADVCL], arg_type),
		ArgumentValue.POSSING: lambda arg_type: _get_ing_maps([WordRelation.ACL, WordRelation.ADVCL], arg_type, possessive=True),

		ArgumentValue.WHAT_S: lambda arg_type: _get_what_s_maps([WordRelation.ACL], arg_type),
		ArgumentValue.WHETHER_S: lambda arg_type: _get_whether_s_maps([WordRelation.ACL], arg_type),
		ArgumentValue.WHERE_S: lambda arg_type: _get_where_s_maps([WordRelation.ACL], arg_type),
		ArgumentValue.WHEN_S: lambda arg_type: _get_when_s_maps([WordRelation.ACL], arg_type),
		ArgumentValue.HOW_MANY_S: lambda arg_type: _get_how_many_s_maps([WordRelation.ACL, WordRelation.NMOD], arg_type),
		ArgumentValue.HOW_MUCH_S: lambda arg_type: _get_how_much_s_maps([WordRelation.ACL, WordRelation.NMOD], arg_type),
		ArgumentValue.HOW_S: lambda arg_type: _get_how_s_maps([WordRelation.ACL], arg_type),
		ArgumentValue.HOW_TO_INF: lambda arg_type: _get_how_s_maps([WordRelation.ACL], arg_type, untensed_only=True),
	}
}


PREP_RELATIONS = {
	ArgumentType.SUBJ: [WordRelation.CASE],
	ArgumentType.OBJ: [WordRelation.CASE],
	ArgumentType.IND_OBJ: [WordRelation.CASE],
	ArgumentType.NP: [WordRelation.CASE],
	ArgumentType.PP: [WordRelation.CASE],
	ArgumentType.PP1: [WordRelation.CASE],
	ArgumentType.PP2: [WordRelation.CASE],
	ArgumentType.MODIFIER: [WordRelation.CASE, WordRelation.MARK, WordRelation.ADVMOD],
	ArgumentType.ING: [WordRelation.MARK],
	ArgumentType.TO_INF: [WordRelation.MARK],
	ArgumentType.SBAR: [WordRelation.MARK],
}


NP_ING_COMPLEX = lambda arg_type1, arg_type2: {
	LexiconType.VERB: lambda preps:
	_get_maps_with_preps(arg_type2, preps, _expand_constraints_separately(
			_get_ing_maps([WordRelation.CCOMP, WordRelation.XCOMP], arg_type2),
			_get_np_maps([WordRelation.NSUBJ], arg_type1))) +
	_expand_constraints_separately(
			_get_ing_maps([WordRelation.CCOMP, WordRelation.XCOMP], arg_type2),
			_get_maps_with_preps(arg_type1, preps, _get_np_maps([WordRelation.NSUBJ], arg_type1))) +
	_get_maps_with_preps(arg_type1, preps, _expand_constraints_separately(
			_get_np_maps([WordRelation.DOBJ], arg_type1),
			_get_ing_maps([WordRelation.ACL], arg_type2))),
	LexiconType.NOUN: lambda preps:
	_get_maps_with_preps(arg_type2, preps, _expand_constraints_separately(
			_get_ing_maps([WordRelation.ACL, WordRelation.ADVCL, WordRelation.XCOMP], arg_type2),
			_get_np_maps([WordRelation.NSUBJ], arg_type1))) +
	_expand_constraints_separately(
			_get_ing_maps([WordRelation.ACL, WordRelation.ADVCL, WordRelation.XCOMP], arg_type2),
			_get_maps_with_preps(arg_type1, preps, _get_np_maps([WordRelation.NSUBJ], arg_type1))) +
	_get_maps_with_preps(arg_type1, preps, _expand_constraints_separately(
			_get_np_maps([WordRelation.NMOD], arg_type1),
			_get_ing_maps([WordRelation.ACL], arg_type2)))
}

P_NP_ING_COMPLEX = lambda preps: \
	_get_maps_with_preps(ArgumentType.PP, preps, _expand_constraints_separately(
		_get_np_maps([WordRelation.NMOD], ArgumentType.PP),
		_get_ing_maps([WordRelation.ACL], ArgumentType.ING))) + \
	_expand_constraints_separately(
		_get_ing_maps([WordRelation.ACL], ArgumentType.ING),
		_get_maps_with_preps(ArgumentType.PP, preps, (_get_np_maps([WordRelation.NSUBJ], ArgumentType.PP)))) + \
	_get_maps_with_preps(ArgumentType.PP, preps, _expand_constraints_separately(
		_get_ing_maps([WordRelation.ACL], ArgumentType.ING),
		_get_np_maps([WordRelation.NSUBJ], ArgumentType.PP)))

COMPLEX_ARG_CONSTRAINTS = {
	(ArgumentType.NP, ArgumentType.ING): {
		(ArgumentValue.NP, ArgumentValue.ING): NP_ING_COMPLEX(ArgumentType.NP, ArgumentType.ING)
	},
	(ArgumentType.OBJ, ArgumentType.ING): {
		(ArgumentValue.NP, ArgumentValue.ING): NP_ING_COMPLEX(ArgumentType.OBJ, ArgumentType.ING)
	},
	(ArgumentType.PP, ArgumentType.ING): {
		(ArgumentValue.NP, ArgumentValue.ING): {
			LexiconType.VERB: P_NP_ING_COMPLEX,
			LexiconType.NOUN: P_NP_ING_COMPLEX
		}
	}
}

SUBCAT_COMPLEX_ARG_COMBINATIONS = {
	SubcatType.NOM_NP_ING: [(ArgumentType.NP, ArgumentType.ING)],
	SubcatType.NOM_NP_ING_OC: [(ArgumentType.OBJ, ArgumentType.ING)],
	SubcatType.NOM_P_NP_ING: [(ArgumentType.PP, ArgumentType.ING)],
	SubcatType.NOM_NP_P_NP_ING: [(ArgumentType.PP, ArgumentType.ING)]
}


def _get_with_more_constraints(constraints_map: ConstraintsMap, relatives_constraints: ORConstraintsMaps) -> ConstraintsMap:
	constraints_map = deepcopy(constraints_map)
	constraints_map.relatives_constraints += relatives_constraints
	return constraints_map


def _choose_relations_for_prep(
		word_relations: Optional[List[WordRelation]] = None,
		arg_type: Optional[ArgumentType] = None
) -> List[WordRelation]:
	if word_relations is not None:
		return word_relations

	if arg_type is not None:
		return PREP_RELATIONS[arg_type]

	raise NotImplementedError()


def _choose_postags_for_prep(
		prep: str, postags: Optional[List[POSTag]] = None,
		arg_type: Optional[ArgumentType] = None
) -> List[POSTag]:
	postags = postags if postags is not None else []
	if prep == TO_PREP:
		postags += [POSTag.TO]
	elif arg_type == ArgumentType.MODIFIER and prep == AS_PREP:
		postags += [POSTag.RB]

	return postags


def _get_constraints_with_one_worded_prep(
		constraints_map: ConstraintsMap, arg_type: ArgumentType,
		prep: str, prep_arg_type: Optional[ArgumentType] = None,
		postags: Optional[List[POSTag]] = None, word_relations: Optional[List[WordRelation]] = None
) -> ConstraintsMap:
	return _get_with_more_constraints(
		constraints_map,
		[ConstraintsMap(
			word_relations=_choose_relations_for_prep(word_relations, arg_type),
			values=[prep],
			postags=_choose_postags_for_prep(prep, postags, arg_type),
			arg_type=prep_arg_type
		)]
	)


def _get_constraints_with_one_worded_preps(
		constraints_map: ConstraintsMap, arg_type: ArgumentType,
		preps: List[str], prep_arg_type: Optional[ArgumentType] = None,
		postags: Optional[List[POSTag]] = None, word_relations: Optional[List[WordRelation]] = None
) -> ORConstraintsMaps:
	postags = postags if postags is not None else []
	ing_preps, to_preps, as_preps, other_preps = [], [], [], []

	for prep in preps:
		if prep == TO_PREP:
			to_preps.append(prep)
		elif prep == AS_PREP:
			as_preps.append(prep)
		elif prep.endswith("ing"):
			ing_preps.append(prep)
		else:
			other_preps.append(prep)

	constraints_maps = []

	if len(other_preps) > 0:
		constraints_maps.append(_get_with_more_constraints(
			constraints_map,
			[ConstraintsMap(
				word_relations=_choose_relations_for_prep(word_relations, arg_type),
				values=other_preps,
				postags=postags,
				arg_type=prep_arg_type
			)]
		))

	if len(to_preps) > 0:
		constraints_maps.append(_get_constraints_with_one_worded_prep(
			constraints_map, arg_type, TO_PREP, prep_arg_type, postags, word_relations)
		)

	if len(as_preps) > 0:
		constraints_maps.append(_get_constraints_with_one_worded_prep(
			constraints_map, arg_type, AS_PREP, prep_arg_type, postags, word_relations)
		)

	if len(ing_preps) > 0:
		modified_constraints_map = deepcopy(constraints_map)
		modified_constraints_map.word_relations = [WordRelation.CCOMP]
		constraints_maps.append(ConstraintsMap(
			word_relations=constraints_map.word_relations,
			values=ing_preps,
			postags=postags + [POSTag.VBG],
			arg_type=prep_arg_type,
			relatives_constraints=[modified_constraints_map]
		))

	return constraints_maps


def _get_constraints_with_two_words_prep(
		constraints_map: ConstraintsMap, arg_type: ArgumentType,
		prep: str, prep_arg_type: Optional[ArgumentType] = None
):
	words = prep.split()
	second_word_constraint = _get_constraints_with_one_worded_prep(
		constraints_map=constraints_map,
		arg_type=arg_type,
		prep=words[1],
		prep_arg_type=prep_arg_type,
		postags=_choose_postags_for_prep(words[1], [POSTag.IN], arg_type)
	)

	return [
		ConstraintsMap(
			word_relations=[WordRelation.ADVMOD],
			values=[words[0]],
			relatives_constraints=[deepcopy(second_word_constraint)],
			arg_type=prep_arg_type
		),
		_get_constraints_with_one_worded_prep(
			constraints_map=deepcopy(second_word_constraint),
			arg_type=arg_type,
			prep=words[0],
			word_relations=[WordRelation.CASE, WordRelation.MARK],
			postags=_choose_postags_for_prep(words[0], [POSTag.IN], arg_type),
			prep_arg_type=prep_arg_type
		),
		_get_with_more_constraints(
			constraints_map,
			relatives_constraints=[ConstraintsMap(
				word_relations=[WordRelation.CASE, WordRelation.MARK],
				values=[words[0]],
				arg_type=prep_arg_type,
				relatives_constraints=[ConstraintsMap(
					word_relations=[WordRelation.MWE],
					values=[words[1]],
					arg_type=prep_arg_type
				)],
			)]
		)
	]


def _get_constraints_with_three_words_prep(
		constraints_map: ConstraintsMap, arg_type: ArgumentType,
		prep: str, prep_arg_type: Optional[ArgumentType] = None
):
	constraints_map_copy = deepcopy(constraints_map)
	constraints_map_copy.word_relations += [WordRelation.ACL]

	words = prep.split()
	return [
		ConstraintsMap(
			word_relations=[WordRelation.NMOD],
			values=[words[1]],
			arg_type=prep_arg_type,
			relatives_constraints=[
				_get_constraints_with_one_worded_prep(
					constraints_map_copy, arg_type, words[2], prep_arg_type,
					_choose_postags_for_prep(words[2], [POSTag.IN], arg_type)
				),
				ConstraintsMap(
					word_relations=[WordRelation.CASE, WordRelation.MARK],
					values=[words[0]], arg_type=prep_arg_type)
			]
		)
	]


def _get_preps_by_n_words(preps: List[str]) -> Dict[int, List[str]]:
	preps_by_n_words = defaultdict(list)
	for prep in preps:
		n_words = len(prep.split())

		if n_words not in [1, 2, 3]:
			raise NotImplementedError()

		preps_by_n_words[n_words].append(prep)

	return preps_by_n_words


def _get_constraints_with_preps(
		constraints_map: ConstraintsMap, arg_type: ArgumentType,
		preps: List[str], prep_arg_type: Optional[ArgumentType] = None
) -> ORConstraintsMaps:
	if len(preps) == 0:
		return [constraints_map]

	preps_by_n_words = _get_preps_by_n_words(preps)
	new_contrainsts_maps = []

	if 1 in preps_by_n_words:
		new_contrainsts_maps += _get_constraints_with_one_worded_preps(
			constraints_map,
			arg_type,
			preps_by_n_words.get(1),
			prep_arg_type,
			[POSTag.IN],
		)

	for prep in preps_by_n_words[2]:
		new_contrainsts_maps += _get_constraints_with_two_words_prep(constraints_map, arg_type, prep, prep_arg_type)

	for prep in preps_by_n_words[3]:
		new_contrainsts_maps += _get_constraints_with_three_words_prep(constraints_map, arg_type, prep, prep_arg_type)

	return new_contrainsts_maps


def _get_maps_with_preps(
		arg_type: ArgumentType, preps: List[str], constraints_maps: ORConstraintsMaps
) -> ORConstraintsMaps:
	prep_arg_type = arg_type if ArgumentType.is_pp_arg(arg_type) else None
	return list(chain(*[_get_constraints_with_preps(c, arg_type, preps, prep_arg_type) for c in constraints_maps]))

#
# def _expand_maps_with_properties(
# 		constraints_maps: ORConstraintsMaps, subcat_type: SubcatType, arg_type: ArgumentType, attributes: List[str]
# ):
# 	is_plural = get_plural_property(subcat_type, arg_type)
# 	is_subjunct = get_subjunct_property(subcat_type, arg_type)
# 	controlled_args = get_controlled_args(subcat_type, arg_type)
#
# 	for constraints_map in constraints_maps:
# 		constraints_map.is_plural = is_plural
# 		constraints_map.is_subjunct = is_subjunct
# 		constraints_map.controlled = controlled_args
# 		constraints_map.attributes = attributes
#

def get_arg_constraints_maps(
		arg_type: ArgumentType, arg_value: ArgumentValue, lexicon_type: LexiconType,
		preps: List[str], is_required: bool
) -> ORConstraintsMaps:
	if arg_type == ArgumentType.PART:
		constraints_maps = _get_particle_maps(preps, arg_type)
		# _expand_maps_with_properties(constraints_maps, subcat_type, arg_type, attributes)
	else:
		constraints_table = PREPOSITIONAL_ARG_CONSTRAINTS if preps else ARG_CONSTRAINTS
		constraints_maps = constraints_table.get(lexicon_type, {}).get(arg_value, lambda _: [])(arg_type)
		constraints_maps = _get_maps_with_preps(arg_type, preps, constraints_maps)

		# if ArgumentType.is_pp_arg(arg_type):
		# 	constraints_maps = _get_maps_with_preps(constraints_maps, arg_type, preps, arg_type)
		# 	_expand_maps_with_properties(constraints_maps, subcat_type, arg_type, attributes)
		# else:
		# 	_expand_maps_with_properties(constraints_maps, subcat_type, arg_type, attributes)
		# 	constraints_maps = _get_maps_with_preps(constraints_maps, arg_type, preps)

	for constraints_map in constraints_maps:
		constraints_map.required = is_required

	return deepcopy(constraints_maps)


def get_maps_with_complex_constraints(
		subcat_type: SubcatType, maps_by_arg: Dict[ArgumentType, ConstraintsMap],
		value_by_arg: Dict[ArgumentType, ArgumentValue],
		preps_by_arg: Dict[ArgumentType, List[str]],
		lexicon_type: LexiconType,
) -> List[ANDConstraintsMaps]:
	complex_combinations = []
	for arg_combination in SUBCAT_COMPLEX_ARG_COMBINATIONS.get(subcat_type, []):
		for value_combination, complex_constraints in COMPLEX_ARG_CONSTRAINTS[arg_combination].items():
			found_mismatch = False
			preps = []

			for arg_type, arg_value in zip(arg_combination, value_combination):
				if value_by_arg.get(arg_type) != arg_value:
					found_mismatch = True
					break

				assert arg_type in preps_by_arg
				preps += preps_by_arg[arg_type]

			if found_mismatch:
				continue

			for constraint in deepcopy(complex_constraints[lexicon_type](preps)):
				constraint.required = True
				complex_combinations.append(
					[constraint] + [m for arg_type, m in maps_by_arg.items() if arg_type not in arg_combination]
				)

	return complex_combinations
