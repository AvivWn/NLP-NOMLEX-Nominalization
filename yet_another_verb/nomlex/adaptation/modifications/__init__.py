from yet_another_verb.nomlex.adaptation.modifications.subcat_not_constraints import \
	get_extra_not_constraints
from yet_another_verb.nomlex.adaptation.modifications.subcat_name_typos import get_correct_subcat_type
from yet_another_verb.nomlex.adaptation.modifications.subcat_requirements import \
	get_default_requires, get_default_optionals
from yet_another_verb.nomlex.adaptation.modifications.nom_type_to_arg_type import \
	get_as_arg_type, get_compatible_args
from yet_another_verb.nomlex.adaptation.modifications.arg_renamings import get_argument_renamings
from yet_another_verb.nomlex.adaptation.modifications.arg_default_preps import \
	get_default_preps_by_arg
from yet_another_verb.nomlex.adaptation.modifications.arg_default_values import get_default_arg_values
from yet_another_verb.nomlex.adaptation.modifications.arg_constraints import get_arg_constraints_maps
from yet_another_verb.nomlex.adaptation.modifications.arg_properties import get_plural_property, \
	get_controlled_args, get_subjunct_property, get_arg_attributes_property