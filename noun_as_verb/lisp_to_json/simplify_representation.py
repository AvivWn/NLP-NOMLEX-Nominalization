import re
from copy import deepcopy

from .lexicon_modifications import argument_constraints, subcat_constraints
from .utils import get_right_value, get_current_specs, is_known, curr_specs
from noun_as_verb.utils import difference_list, list_to_regex
from noun_as_verb.constants.lexicon_constants import *
from noun_as_verb.constants.ud_constants import *

# For debug
missing_required = []
args_without_pos = []

def split_positions(subcat, complement_type, argument_positions):
	subcat[complement_type].update({ARG_POSITIONS: [], ARG_PREFIXES: [], ARG_CONSTRAINTS: []})

	# Split the possible positions into different argument properties
	for argument_position in argument_positions:
		if argument_position.islower():
			subcat[complement_type][ARG_PREFIXES] += [argument_position]
		elif is_known(argument_position, ["POS"], "POS"):
			subcat[complement_type][ARG_POSITIONS] += [argument_position]
		else:
			raise Exception(f"Unknown complement positon ({get_current_specs()}).")

	# Delete any complement that don't get prefixes positions and any position
	if subcat[complement_type][ARG_POSITIONS] == [] and subcat[complement_type][ARG_PREFIXES] == []:
		del subcat[complement_type]
		return

	# Prefixes can be also the position itself
	if subcat[complement_type][ARG_PREFIXES] != []:
		# Only "direct" relations to the predicate can get a "prefix" position
		if complement_type not in [COMP_FOR_TO_INF, COMP_PART]:
			subcat[complement_type][ARG_POSITIONS] += [POS_PREFIX]

def simplify_complement_positions(subcat, complement_type):
	"""
	Simplifies the representation of the given complement type in the given subcat
	:param subcat: a dictionary of the subcategorization info
	:param complement_type: the type of the complement that is needed to be simplified
	:return: None
	"""

	tmp_subcat = deepcopy(subcat)

	# Ignore unkown types of complements and subcat constraints
	if not is_known(complement_type, ["COMP", "SUBCAT_CONSTRAINT"], "SUBCAT COMPLEMENTS & CONSTRAINTS"):
		subcat.pop(complement_type)
		return

	# A complement with string value belongs to the constraints list
	if type(tmp_subcat[complement_type]) == str:
		subcat[SUBCAT_CONSTRAINTS].append(complement_type)
		subcat.pop(complement_type)

		if tmp_subcat[complement_type] != "T":
			raise Exception(f"Unknown constraint value- {tmp_subcat[complement_type]} ({get_current_specs()}).")

		return

	# Otherwise, dictionary
	subcat[complement_type] = {}

	# Split the possible positions into 3 different positions
	split_positions(subcat, complement_type, tmp_subcat.get(complement_type, []))

def simplify_representation(subcat, subcat_type, is_verb=False):
	"""
	Simplifies representation of the given subcat
	:param subcat: a dictionary of the subcategorization info ({ARG1: [POS1, POS2, ...], ARG2: [...], NOT: [...], REQUIRED: [...], OPTIONALS: [...]})
	:param subcat_type: the type of the subcategorization
	:param is_verb: whether or not the given subcat is for verb rearranging (otherwise- nominalization)
	:return: None
	"""

	# Requires that their possitions are missing, should be optionals
	for complement_type in deepcopy(subcat[SUBCAT_REQUIRED]):
		curr_specs["comp"] = complement_type

		if complement_type not in subcat.keys():
			subcat[SUBCAT_REQUIRED].remove(complement_type)
			subcat[SUBCAT_OPTIONAL].append(complement_type)
			missing_required.append(complement_type)

	curr_specs["comp"] = None
	tmp_subcat = deepcopy(subcat)
	subcat.pop(ARG_CONSTRAINT_DET_POSS_NO_OTHER_OBJ)
	subcat.pop(ARG_CONSTRAINT_N_N_MOD_NO_OTHER_OBJ)

	all_complements = deepcopy(list(set(difference_list(subcat.keys(), [SUBCAT_OPTIONAL, SUBCAT_REQUIRED, SUBCAT_NOT]) + subcat[SUBCAT_REQUIRED])))
	subcat[SUBCAT_CONSTRAINTS] = []

	# Get more arguments constraints
	more_argument_constraints = get_right_value(argument_constraints, subcat_type, {}, is_verb)

	# Rearrange the subentry for any possible complement for this subcat
	for complement_type in all_complements:
		curr_specs["comp"] = complement_type

		simplify_complement_positions(subcat, complement_type)

		if complement_type not in subcat.keys():
			continue

		arg = subcat[complement_type]

		# Update more manual constraints for that compelement/argument
		arg.update(more_argument_constraints.get(complement_type, {}))

		# Update the possible root postags for specific complements
		if ARG_ROOT_UPOSTAGS not in arg.keys():
			if complement_type == COMP_PART:
				arg[ARG_ROOT_UPOSTAGS] = [UPOS_PART, UPOS_ADP]
			elif complement_type in [COMP_IND_OBJ, COMP_OBJ, COMP_SUBJ, COMP_SECOND_SUBJ, COMP_NP, COMP_AS_NP_OC, COMP_AS_NP_SC]:
				arg[ARG_ROOT_UPOSTAGS] = [UPOS_PROPN, UPOS_NOUN, UPOS_PRON, UPOS_DET]
			elif complement_type in [COMP_PP, COMP_PP1, COMP_PP2]: # VERB is not allowed for PP head
				arg[ARG_ROOT_UPOSTAGS] = [UPOS_PROPN, UPOS_NOUN, UPOS_PRON, UPOS_DET]
			else:
				args_without_pos.append(complement_type)

		# Add the DET_POSS_NO_OTHER_OBJ\N_N_MOD_NO_OTHER_OBJ constriants for the nominalization (if it is relevant)
		if not is_verb and complement_type in tmp_subcat[ARG_CONSTRAINT_DET_POSS_NO_OTHER_OBJ]:
			arg[ARG_CONSTRAINTS] += [ARG_CONSTRAINT_DET_POSS_NO_OTHER_OBJ]

		if not is_verb and complement_type in tmp_subcat[ARG_CONSTRAINT_N_N_MOD_NO_OTHER_OBJ]:
			arg[ARG_CONSTRAINTS] += [ARG_CONSTRAINT_N_N_MOD_NO_OTHER_OBJ]

		# Gerund phrase cannot start with WH words or "that"
		if complement_type.startswith("ING") or complement_type.startswith("POSSING"):
			arg[ARG_ILLEGAL_PREFIXES] = WHERE_WHEN_OPTIONS + HOW_OPTIONS + WH_VERB_OPTIONS + ["that"]

		# Infinitival phrases cannot start with the preposition "for" (cause it refers to FOR-TO-INF)
		elif complement_type.startswith("TO-INF"):
			arg[ARG_ILLEGAL_PREFIXES] = ["for"]

		elif complement_type == COMP_HOW_S:
			arg[ARG_ILLEGAL_PREFIXES] = []
			# Update illegal prefixes to include a preposition as a prefix (if it is relevant)
			for prefix in arg[ARG_PREFIXES]:
				just_preposition_prefix = re.sub(list_to_regex(WHERE_WHEN_OPTIONS + WH_VERB_OPTIONS + HOW_TO_OPTIONS + HOW_OPTIONS, "|"), '', prefix).strip()
				arg[ARG_ILLEGAL_PREFIXES] += [(just_preposition_prefix + " " + illegal_prefix).strip() for illegal_prefix in ["how to", "how much", "how many"]]

		elif complement_type.startswith("AS-") and complement_type != COMP_AS_IF_S_SUBJUNCT:
			arg[ARG_ILLEGAL_PREFIXES] = ["as if"]

		if complement_type == COMP_PP and ARG_CONSTRAINT_REQUIRED_PREFIX in arg[ARG_CONSTRAINTS]:
			arg[ARG_CONSTRAINTS].remove(ARG_CONSTRAINT_REQUIRED_PREFIX)

		if complement_type == COMP_PART:
			arg[ARG_CONSTRAINTS] += [ARG_CONSTRAINT_REQUIRED_PREFIX]

	curr_specs["comp"] = None

	# Update more manual constraint for this subcategorization
	more_subcat_constraints = get_right_value(subcat_constraints, subcat_type, [], is_verb)
	subcat[SUBCAT_CONSTRAINTS] += more_subcat_constraints