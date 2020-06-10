from .argument import *

class Subcat:
	arguments = defaultdict(Argument)
	requires: list
	optionals: list
	constraints: list
	not_constraints: dict
	is_verb: bool

	def __init__(self, subcat_info: dict, is_verb):
		self.arguments = defaultdict(Argument)
		for complement_type in difference_list(subcat_info.keys(), [SUBCAT_REQUIRED, SUBCAT_OPTIONAL, SUBCAT_NOT, SUBCAT_CONSTRAINTS]):
			self.arguments[complement_type] = Argument(subcat_info[complement_type], is_verb)

		self.requires = subcat_info.get(SUBCAT_REQUIRED, [])
		self.optionals = difference_list(self.arguments.keys(), subcat_info.get(SUBCAT_REQUIRED, []))
		self.not_constraints = subcat_info.get(SUBCAT_NOT, [])
		self.constraints = subcat_info.get(SUBCAT_CONSTRAINTS, [])
		self.is_verb = is_verb

	def check_uniqueness(self):
		return True

	def check_order(self):
		return True

	def check_only_no_other(self):
		return True

	def check_arguments_compatibility(self, args_per_candidates, dependency_tree, argument_types, argument_candidates):
		#linked_arguments = []

		#print(argument_types)

		for complement_type in argument_types:
			argument = self.arguments[complement_type]

			# if argument.is_linked():
			# 	linked_arguments.append(argument)

			for candidate_index in argument_candidates:
				if argument.check_constraints(dependency_tree, candidate_index):
					args_per_candidates[candidate_index].append(complement_type)

		# for complement_type in linked_arguments:
		# 	argument = self.arguments[complement_type]
		#
		# 	for the_linked_arg in argument.get_linked():
		# 		cadidates = self.get_argument_candidates(dependency_tree, the_linked_arg)
		#
		# 		for candidate_index in cadidates:
		# 			if argument.check_constraints(dependency_tree, candidate_index):
		# 				args_per_candidates[candidate_index].append(complement_type)

		#print(1, args_per_candidates)

	def check_constraints(self, option):
		if not set(self.requires).issubset(option.keys()):
			return False

		if SUBCAT_CONSTRAINT_ADVP_OR_ADJP in self.constraints:
			if COMP_ADVP not in option and COMP_ADJP not in option:
				return False

		return True

	def extract_linked_arguments(self, dependency_tree: list, linked_arguments, extraction):
		for complement_type in difference_list(linked_arguments, extraction.keys()):
			linked_argument = self.arguments[complement_type]
			linked_argument.handle_linked_argument(dependency_tree, complement_type, extraction)


	def extract_arguments(self, dependency_tree: list, argument_candidates: list):
		args_per_candidates = defaultdict(list)

		linked_arguments = [complement_type for complement_type in self.arguments if self.arguments[complement_type].linked_positions != {}]
		#print("here", linked_arguments)

		# if linked_arguments != []:
		# 	1 / 0
		# 	exit()

		# Check the compatability of the candidates with the required arguments first
		self.check_arguments_compatibility(args_per_candidates, dependency_tree, self.requires, argument_candidates)
		if len(args_per_candidates.keys()) < len(difference_list(self.requires, linked_arguments)):
			return []

		# Then, check for the optional arguments
		self.check_arguments_compatibility(args_per_candidates, dependency_tree, self.optionals, argument_candidates)

		#print("args_per_candidate", args_per_candidates)

		# Retrieve all the possible extractions of the candidate arguments and sort it by the number of arguments
		extractions = [dict(zip(args_per_candidates, v)) for v in product(*args_per_candidates.values()) if len(list(v)) == len(set(v))]
		extractions = sorted(extractions, key=lambda k: len(k.keys()), reverse=True)

		#print(extractions)

		# Get the correct extractions
		# meaning the ines that fulfills the constraints of the subcat
		correct_extractions = []
		for extraction in extractions:
			extraction = reverse_dict(extraction)
			self.extract_linked_arguments(dependency_tree, linked_arguments, extraction)
			#print(extraction)

			# Did we found an extraction that includes that extraction?
			is_sub_extraction = any([extraction.items() <= other_extraction.items() for other_extraction in correct_extractions])
			#print("111111", extraction, correct_extractions)

			# Check constraints on that extraction
			if not is_sub_extraction and self.check_constraints(extraction):
				correct_extractions.append(extraction)

		#print("HERE", correct_extractions)
		return correct_extractions