from .lexicon_modifications import *
from .utils import *

# For debug
missing_required = []
args_without_pos = []

def split_positions(subcat, complement_type, argument_positions, referenced_arg):
	subcat[complement_type].update({referenced_arg: {ARG_CONSTANTS: [], ARG_PREFIXES: [], ARG_CONSTRAINTS: []}})

	# Split the possible positions into 3 different positions
	for argument_position in argument_positions:
		if type(argument_position) == dict:
			for new_referenced_arg, positions in argument_position.items():
				split_positions(subcat, complement_type, positions, new_referenced_arg)
		elif argument_position.islower():
			subcat[complement_type][referenced_arg][ARG_PREFIXES] += [argument_position]
		elif is_known(argument_position, ["POS"], "POS"):
			subcat[complement_type][referenced_arg][ARG_CONSTANTS] += [argument_position]
		else:
			raise Exception(f"Unknown complement positon ({get_current_specs()}).")

def simplify_complement_positions(subcat, complement_type, is_verb=False):
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
	standard_referenced = get_linked_arg(is_verb)
	split_positions(subcat, complement_type, tmp_subcat.get(complement_type, []), standard_referenced)

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

		simplify_complement_positions(subcat, complement_type, is_verb)

		if complement_type in subcat.keys():
			for complement_by_referenced in subcat[complement_type].values():

				# Update more manual constraints for that compelement/argument
				complement_by_referenced.update(more_argument_constraints.get(complement_type, {}))

				# Update the possible root postags for specific complements
				if ARG_ROOT_UPOSTAGS not in complement_by_referenced.keys():
					if complement_type == COMP_PART:
						complement_by_referenced[ARG_ROOT_UPOSTAGS] = [UPOS_PART]
					elif complement_type in [COMP_IND_OBJ, COMP_FOR_NP, COMP_P_IND_OBJ, COMP_PP, COMP_PP1, COMP_PP2, COMP_OBJ, COMP_SUBJ, COMP_NP, COMP_AS_NP_OC, COMP_AS_NP_SC]:
						complement_by_referenced[ARG_ROOT_UPOSTAGS] = [UPOS_PROPN, UPOS_NOUN, UPOS_PRON]
					else:
						args_without_pos.append(complement_type)

				# Add the DET_POSS_NO_OTHER_OBJ\N_N_MOD_NO_OTHER_OBJ constriants for the nominalization (if it is relevant)
				if not is_verb and complement_type in tmp_subcat[ARG_CONSTRAINT_DET_POSS_NO_OTHER_OBJ]:
					complement_by_referenced[ARG_CONSTRAINTS] += [ARG_CONSTRAINT_DET_POSS_NO_OTHER_OBJ]

				if not is_verb and complement_type in tmp_subcat[ARG_CONSTRAINT_N_N_MOD_NO_OTHER_OBJ]:
					complement_by_referenced[ARG_CONSTRAINTS] += [ARG_CONSTRAINT_N_N_MOD_NO_OTHER_OBJ]

	curr_specs["comp"] = None

	# Update more manual constraint for this subcategorization
	more_subcat_constraints = get_right_value(subcat_constraints, subcat_type, [], is_verb)
	subcat[SUBCAT_CONSTRAINTS] += more_subcat_constraints