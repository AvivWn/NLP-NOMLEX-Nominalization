from typing import List, Dict
from copy import deepcopy
from itertools import chain
from collections import defaultdict

from yet_another_verb.nomlex.constants import LexiconType, ArgumentValue, ArgumentType, \
	WordRelation, POSTag, SubcatType
from yet_another_verb.nomlex.constants.word_postag import NOUN_POSTAGS, VERB_POSTAGS, \
	ADVERB_POSTAGS, ADJECTIVE_POSTAGS
from yet_another_verb.nomlex.representation.constraints_map import ConstraintsMap
from yet_another_verb.nomlex.adaptation.modifications.arg_properties import get_plural_property, \
	get_subjunct_property, get_controlled_args, get_arg_attributes_property

ConstraintsMaps = List[ConstraintsMap]  # (constraint 1) OR (constraint 2) OR (...)


def _exapnd_constraint(constraint_map: ConstraintsMap, relatives_constraints: ConstraintsMaps) -> ConstraintsMaps:
	if len(relatives_constraints) == 0:
		return [constraint_map]

	resulted_maps = []
	for relative_constraints in relatives_constraints:
		expanded_consrtaint = deepcopy(constraint_map)
		expanded_consrtaint.relatives_constraints.append(relative_constraints)
		resulted_maps.append(expanded_consrtaint)

	return resulted_maps


def _get_mark_map(values: List[str], postags: List[POSTag] = None) -> ConstraintsMap:
	postags = [] if postags is None else postags
	return ConstraintsMap(word_relations=[WordRelation.MARK], values=values, postags=postags)


def _get_advmod_map(values: List[str], postags: List[POSTag] = None) -> ConstraintsMap:
	postags = [] if postags is None else postags
	return ConstraintsMap(word_relations=[WordRelation.ADVMOD], values=values, postags=postags)


def _get_np_maps(word_relations: List[WordRelation]) -> ConstraintsMaps:
	return [ConstraintsMap(word_relations=word_relations, postags=NOUN_POSTAGS)]


def _get_possessive_maps() -> ConstraintsMaps:
	return [
		ConstraintsMap(word_relations=[WordRelation.NMOD_POSS], postags=NOUN_POSTAGS),
		ConstraintsMap(word_relations=[WordRelation.NSUBJ], postags=[POSTag.PRP_POSS]),
		ConstraintsMap(word_relations=[WordRelation.NSUBJ], postags=NOUN_POSTAGS, relatives_constraints=[
			ConstraintsMap(values=["'s"], postags=[POSTag.POS])
		])
	]


def _get_adjective_maps(word_relations: List[WordRelation]) -> ConstraintsMaps:
	return [ConstraintsMap(word_relations=word_relations, postags=ADJECTIVE_POSTAGS)]


def _get_adverb_maps():
	return [ConstraintsMap(word_relations=[WordRelation.ADVMOD], postags=ADVERB_POSTAGS)]


def _get_to_inf_maps(word_relations: List[WordRelation]) -> ConstraintsMaps:
	return [ConstraintsMap(
		word_relations=word_relations,
		postags=[POSTag.VB],
		relatives_constraints=[_get_mark_map(["to"], [POSTag.TO])]
	), ConstraintsMap(
		word_relations=word_relations,
		relatives_constraints=[
			_get_mark_map(["to"], [POSTag.TO]),
			ConstraintsMap(
				word_relations=[WordRelation.COP],
				postags=[POSTag.VB],
				values=["be"]
			)
		]
	)]


def _get_ing_maps(word_relations: List[WordRelation], possessive: bool = False) -> ConstraintsMaps:
	possessive_constraints = [] if not possessive else _get_possessive_maps()

	return _exapnd_constraint(
		ConstraintsMap(
			word_relations=word_relations,
			postags=[POSTag.VBG],
		), possessive_constraints
	) + _exapnd_constraint(
		ConstraintsMap(
			word_relations=word_relations,
			relatives_constraints=[
				ConstraintsMap(
					word_relations=[WordRelation.COP],
					postags=[POSTag.VBG],
					values=["being"]
				)
			]
		), possessive_constraints
	)


def _get_sbar_maps(
		word_relations: List[WordRelation],
		relatives_constraints: ConstraintsMaps = None,
		tensed_only: bool = False,
		untensed_only: bool = False
) -> ConstraintsMaps:
	assert tensed_only ^ untensed_only or (not tensed_only and not untensed_only)

	relatives_constraints = [] if relatives_constraints is None else relatives_constraints

	if tensed_only:
		relatives_constraints += [ConstraintsMap(word_relations=[WordRelation.NMOD_POSS], postags=NOUN_POSTAGS)]

	if untensed_only:
		relatives_constraints += [_get_mark_map(["to"], [POSTag.TO])]

	return [ConstraintsMap(
		word_relations=word_relations,
		postags=[POSTag.VB] if untensed_only else VERB_POSTAGS,
		relatives_constraints=relatives_constraints
	)]


def _get_that_s_maps(word_relations: List[WordRelation]) -> ConstraintsMaps:
	return _get_sbar_maps(word_relations, [_get_mark_map(["that"])], tensed_only=True)


def _get_whether_s_maps(word_relations: List[WordRelation]) -> ConstraintsMaps:
	return _get_sbar_maps(word_relations, [_get_mark_map(["whether"])])


def _get_what_s_maps(word_relations: List[WordRelation]) -> ConstraintsMaps:
	return _get_sbar_maps(word_relations, [ConstraintsMap(
		values=["what"],
		word_relations=[WordRelation.NSUBJ, WordRelation.DOBJ]
	)])


def _get_if_s_maps(word_relations: List[WordRelation]) -> ConstraintsMaps:
	return _get_sbar_maps(word_relations, [_get_mark_map(["if"])])


def _get_where_s_maps(word_relations: List[WordRelation]) -> ConstraintsMaps:
	return _get_sbar_maps(word_relations, [_get_advmod_map(["where"])])


def _get_when_s_maps(word_relations: List[WordRelation]) -> ConstraintsMaps:
	return _get_sbar_maps(word_relations, [_get_advmod_map(["when"])])


def _get_how_much_or_many_s_maps(word_relations: List[WordRelation], much_or_many: str) -> ConstraintsMaps:
	how_much_or_many_s_map = ConstraintsMap(
		word_relations=[WordRelation.ADVMOD],
		values=[much_or_many],
		relatives_constraints=[_get_advmod_map(["how"])]
	)
	return _get_sbar_maps(
		word_relations,
		[ConstraintsMap(
			word_relations=[WordRelation.DOBJ, WordRelation.NSUBJ, WordRelation.NSUBJPASS],
			relatives_constraints=[how_much_or_many_s_map]
		)]
	) + _get_sbar_maps(word_relations, [how_much_or_many_s_map])


def _get_how_much_s_maps(word_relations: List[WordRelation]) -> ConstraintsMaps:
	return _get_how_much_or_many_s_maps(word_relations, "much")


def _get_how_many_s_maps(word_relations: List[WordRelation]) -> ConstraintsMaps:
	return _get_how_much_or_many_s_maps(word_relations, "many")


def _get_how_s_maps(word_relations: List[WordRelation], untensed_only: bool = False) -> ConstraintsMaps:
	return _get_sbar_maps(word_relations, [_get_advmod_map(["how"])], untensed_only=untensed_only)


def _get_as_if_s_maps(word_relations: List[WordRelation]) -> ConstraintsMaps:
	if_map = ConstraintsMap(word_relations=[WordRelation.MWE], values=["if"])
	return _get_sbar_maps(
		word_relations,
		[ConstraintsMap(
			word_relations=[WordRelation.MARK],
			values=["as"],
			relatives_constraints=[if_map]
		)]
	)


def _get_particle_maps(values: List[str]) -> ConstraintsMaps:
	return [ConstraintsMap(word_relations=[WordRelation.PRT, WordRelation.COMPOUND_PRT], values=values, postags=[POSTag.RP])]


ARG_CONSTRAINTS = {
	LexiconType.VERB: {
		ArgumentValue.NSUBJ: _get_np_maps([WordRelation.NSUBJ]),
		ArgumentValue.NSUBJPASS: _get_np_maps([WordRelation.NSUBJPASS]),
		ArgumentValue.DOBJ: _get_np_maps([WordRelation.DOBJ]),
		ArgumentValue.IOBJ: _get_np_maps([WordRelation.IOBJ]),

		ArgumentValue.ADMOD: _get_adverb_maps(),

		ArgumentValue.TO_INF: _get_to_inf_maps([WordRelation.ADVCL, WordRelation.XCOMP]),
		ArgumentValue.ING: _get_ing_maps([WordRelation.XCOMP]),
		ArgumentValue.POSSING: _get_ing_maps([WordRelation.CCOMP], possessive=True),

		ArgumentValue.SBAR: _get_sbar_maps([WordRelation.CCOMP], tensed_only=True),
		ArgumentValue.THAT_S: _get_that_s_maps([WordRelation.CCOMP]),
		ArgumentValue.IF_S: _get_if_s_maps([WordRelation.ADVCL]),
		ArgumentValue.WHAT_S: _get_what_s_maps([WordRelation.CCOMP]),
		ArgumentValue.WHETHER_S: _get_whether_s_maps([WordRelation.CCOMP]),
		ArgumentValue.WHERE_S: _get_where_s_maps([WordRelation.CCOMP, WordRelation.ADVCL]),
		ArgumentValue.WHEN_S: _get_when_s_maps([WordRelation.CCOMP, WordRelation.ADVCL]),
		ArgumentValue.HOW_MANY_S: _get_how_many_s_maps([WordRelation.CCOMP]),
		ArgumentValue.HOW_MUCH_S: _get_how_much_s_maps([WordRelation.CCOMP]),
		ArgumentValue.HOW_S: _get_how_s_maps([WordRelation.CCOMP]),
		ArgumentValue.HOW_TO_INF: _get_how_s_maps([WordRelation.CCOMP], untensed_only=True),
		ArgumentValue.AS_IF_S: _get_as_if_s_maps([WordRelation.ADVCL]),
	},
	LexiconType.NOUN: {
		ArgumentValue.N_N_MOD: _get_np_maps([WordRelation.COMPOUND]),
		ArgumentValue.DET_POSS: _get_possessive_maps(),

		ArgumentValue.ADMOD: _get_adverb_maps(),
		ArgumentValue.AJMOD: _get_adjective_maps([WordRelation.AMOD]),

		ArgumentValue.TO_INF: _get_to_inf_maps([WordRelation.ACL]),
		ArgumentValue.ING: _get_ing_maps([WordRelation.ACL]),
		ArgumentValue.POSSING: _get_ing_maps([WordRelation.ACL], possessive=True),

		ArgumentValue.SBAR: _get_that_s_maps([WordRelation.CCOMP]),
		ArgumentValue.THAT_S: _get_that_s_maps([WordRelation.CCOMP]),
		ArgumentValue.AS_IF_S: _get_as_if_s_maps([WordRelation.ADVCL]),
	}
}


PREPOSITIONAL_ARG_CONSTRAINTS = {
	LexiconType.VERB: {
		ArgumentValue.NP: _get_np_maps([WordRelation.NMOD]),

		ArgumentValue.AJMOD: _get_adjective_maps([WordRelation.ADVCL]),

		ArgumentValue.TO_INF: _get_to_inf_maps([WordRelation.ADVCL]),
		ArgumentValue.ING: _get_ing_maps([WordRelation.ADVCL]),
		ArgumentValue.POSSING: _get_ing_maps([WordRelation.ADVCL], possessive=True),

		ArgumentValue.IF_S: _get_if_s_maps([WordRelation.ADVCL]),
		ArgumentValue.WHAT_S: _get_what_s_maps([WordRelation.ADVCL]),
		ArgumentValue.WHETHER_S: _get_whether_s_maps([WordRelation.ADVCL]),
	},
	LexiconType.NOUN: {
		ArgumentValue.NP: _get_np_maps([WordRelation.NMOD]),

		ArgumentValue.AJMOD: _get_adjective_maps([WordRelation.ACL]),

		ArgumentValue.TO_INF: _get_to_inf_maps([WordRelation.ACL]),
		ArgumentValue.ING: _get_ing_maps([WordRelation.ACL, WordRelation.ADVCL]),
		ArgumentValue.POSSING: _get_ing_maps([WordRelation.ACL, WordRelation.ADVCL], possessive=True),

		ArgumentValue.WHAT_S: _get_what_s_maps([WordRelation.ACL]),
		ArgumentValue.WHETHER_S: _get_whether_s_maps([WordRelation.ACL]),
		ArgumentValue.WHERE_S: _get_where_s_maps([WordRelation.ACL]),
		ArgumentValue.WHEN_S: _get_when_s_maps([WordRelation.ACL]),
		ArgumentValue.HOW_MANY_S: _get_how_many_s_maps([WordRelation.ACL]),
		ArgumentValue.HOW_MUCH_S: _get_how_much_s_maps([WordRelation.ACL]),
		ArgumentValue.HOW_S: _get_how_s_maps([WordRelation.ACL]),
		ArgumentValue.HOW_TO_INF: _get_how_s_maps([WordRelation.ACL], untensed_only=True),
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
	ArgumentType.MODIFIER: [WordRelation.MARK],
	ArgumentType.ING: [WordRelation.MARK],
	ArgumentType.TO_INF: [WordRelation.MARK],
	ArgumentType.SBAR: [WordRelation.MARK],
}


def _get_constraints_with_one_worded_preps(
		constraints_map: ConstraintsMap, arg_type: ArgumentType, preps: List[str],
		postags: List[POSTag] = None, word_relations: List[WordRelation] = None
) -> ConstraintsMap:
	one_worded_map = deepcopy(constraints_map)
	word_relations = word_relations if word_relations is not None else PREP_RELATIONS[arg_type]
	postags = postags if postags is not None else []
	one_worded_map.relatives_constraints += [ConstraintsMap(word_relations=word_relations, values=preps, postags=postags)]
	return one_worded_map


def _get_constraints_with_two_words_prep(
		constraints_map: ConstraintsMap, arg_type: ArgumentType, prep: str
):
	words = prep.split()
	second_word_constraint = _get_constraints_with_one_worded_preps(constraints_map, arg_type, [words[1]])

	return [
		ConstraintsMap(
			word_relations=[WordRelation.ADVMOD],
			values=[words[0]],
			relatives_constraints=[deepcopy(second_word_constraint)]
		),
		_get_constraints_with_one_worded_preps(
			constraints_map=deepcopy(second_word_constraint),
			arg_type=arg_type,
			preps=[words[0]],
			word_relations=[WordRelation.CASE, WordRelation.MARK]
		),
		_get_with_more_constraints(
			constraints_map,
			relatives_constraints=[ConstraintsMap(
				word_relations=[WordRelation.CASE, WordRelation.MARK],
				relatives_constraints=[ConstraintsMap(word_relations=[WordRelation.MWE], values=[words[1]])],
				values=[words[0]]
			)]
		)
	]


def _get_constraints_with_three_words_prep(
		constraints_map: ConstraintsMap, arg_type: ArgumentType, prep: str
):
	words = prep.split()
	return [
		ConstraintsMap(
			word_relations=[WordRelation.NMOD],
			values=[words[1]],
			relatives_constraints=[
				_get_constraints_with_one_worded_preps(constraints_map, arg_type, [words[2]]),
				ConstraintsMap(word_relations=[WordRelation.CASE, WordRelation.MARK], values=[words[0]])
			]
		)
	]


def _get_with_more_constraints(constraints_map: ConstraintsMap, relatives_constraints: ConstraintsMaps) -> ConstraintsMap:
	constraints_map = deepcopy(constraints_map)
	constraints_map.relatives_constraints += relatives_constraints
	return constraints_map


def _get_preps_by_n_words(preps: List[str]) -> Dict[int, List[str]]:
	preps_by_n_words = defaultdict(list)
	for prep in preps:
		n_words = len(prep.split())

		if n_words not in [1, 2, 3]:
			raise NotImplementedError()

		preps_by_n_words[n_words].append(prep)

	return preps_by_n_words


def _get_constraints_with_preps(
		constraints_map: ConstraintsMap, arg_type: ArgumentType, preps: List[str]
) -> ConstraintsMaps:
	if len(preps) == 0:
		return [constraints_map]

	preps_by_n_words = _get_preps_by_n_words(preps)
	new_contrainsts_maps = []

	if 1 in preps_by_n_words:
		if arg_type is ArgumentType.PART:
			new_contrainsts_maps += _get_with_more_constraints(constraints_map, _get_particle_maps(preps))
		else:
			new_contrainsts_maps += [_get_constraints_with_one_worded_preps(
				constraints_map,
				arg_type,
				preps_by_n_words.get(1),
				[POSTag.IN]
			)]

	for prep in preps_by_n_words[2]:
		assert arg_type != ArgumentType.PART
		new_contrainsts_maps += _get_constraints_with_two_words_prep(constraints_map, arg_type, prep)

	for prep in preps_by_n_words[3]:
		assert arg_type != ArgumentType.PART
		new_contrainsts_maps += _get_constraints_with_three_words_prep(constraints_map, arg_type, prep)

	return new_contrainsts_maps


def _update_requirement(constraint_map: ConstraintsMap, is_required: bool):
	constraint_map.required = is_required


def get_arg_constraints_maps(
		subcat_type: SubcatType, arg_type: ArgumentType, arg_value: ArgumentValue, lexicon_type: LexiconType,
		preps: List[str], is_required: bool
) -> ConstraintsMaps:
	attributes_property = get_arg_attributes_property(arg_type)
	attributes = [] if attributes_property is None else attributes_property
	is_plural = get_plural_property(subcat_type, arg_type)
	is_subjunct = get_subjunct_property(subcat_type, arg_type)
	controlled_args = get_controlled_args(subcat_type, arg_type)

	constraints_table = PREPOSITIONAL_ARG_CONSTRAINTS if preps else ARG_CONSTRAINTS
	constraints_maps = deepcopy(constraints_table.get(lexicon_type, {}).get(arg_value, []))
	for constraints_map in constraints_maps:
		constraints_map.arg_type = arg_type
		constraints_map.is_plural = is_plural
		constraints_map.is_subjunct = is_subjunct
		constraints_map.controlled = controlled_args
		constraints_map.attributes = attributes

	constraints_maps = list(chain(*[_get_constraints_with_preps(c, arg_type, preps) for c in constraints_maps]))

	for constraints_map in constraints_maps:
		_update_requirement(constraints_map, is_required)

	return deepcopy(constraints_maps)
