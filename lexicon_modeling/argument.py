from .utils import *

class Argument:
	# Possible positions as two different lists
	constant_positions: list
	prefix_pattern: str

	# Other constraints on the argument
	illegal_prefix_pattern: str
	head_pattern: str
	head_postags: list
	head_links: list
	constraints: list

	is_verb: bool

	def __init__(self, argument_info, is_verb):
		self.constant_positions = argument_info.get(ARG_CONSTANTS, [])
		self.linked_positions = argument_info.get(ARG_LINKED, [])
		self.head_postags = argument_info.get(ARG_ROOT_UPOSTAGS, [])
		self.head_links = argument_info.get(ARG_ROOT_RELATIONS, [])
		self.constraints = argument_info.get(ARG_CONSTRAINTS, [])

		# Translate the patterns list into regex patterns
		self.prefix_pattern = list_to_regex(argument_info.get(ARG_PREFIXES, []), "|", start_constraint="^")
		self.illegal_prefix_pattern = list_to_regex(argument_info.get(ARG_ILLEGAL_PREFIXES, []), "|", start_constraint="^")
		self.head_pattern = list_to_regex(argument_info.get(ARG_ROOT_PATTERNS, []), "|")

		self.is_verb = is_verb

	def check_head(self, head_info):
		return self.head_postags == [] or head_info[WORD_UPOS_TAG] in self.head_postags and \
			   re.search(self.head_pattern, head_info[WORD_TEXT].lower(), re.M)

	def deplink_to_position(self, word_info):
		dep_link = word_info[WORD_DEP_LINK]

		positions = get_right_value(LINK_TO_POS, dep_link, default=[], is_verb=self.is_verb)

		# if POS_PREFIX in positions:
		# 	positions.append(word_info[WORD_SUB_TREE])
		# 	positions.remove(POS_PREFIX)

		return positions

	def check_possessive(self, word_info):
		if ARG_CONSTRAINT_POSSESSIVE in self.constraints:
			if not word_info[WORD_SUB_TREE].lower().endswith("'s") and not word_info[WORD_TEXT].lower() in ["my", "your", "his", "our", "her", "their"]:
				return False

		return True

	def check_links(self, dependency_tree, word_info):
		for word in dependency_tree:

			for head_link in self.head_links:
				head_link_info = head_link.split("_")
				link = head_link_info[0]

				if len(head_link_info) == 2:
					specific_word = head_link_info[1]

					if word[WORD_HEAD_ID] == word_info[WORD_INDEX] and word[WORD_TEXT] == link and word[WORD_TEXT] == specific_word:
						return True

				elif word[WORD_HEAD_ID] == word_info[WORD_INDEX] == link:
					return True

		return False

	def check_constraints(self, dependency_tree, candidate_index, linked=None):
		candidate_info = dependency_tree[candidate_index]

		# Get the "position" type of the candidate (like DET-POSS, of, for and so on)
		candidate_positions = self.deplink_to_position(candidate_info)

		# if linked is not None:
		# 	print(2, self.linked_positions[linked])

		# Check the constraint when that position is not constant
		for candidate_position in candidate_positions:
			is_prefix_legal = True
			is_prefix_illegal = False

			#print(candidate_position)

			if candidate_position == POS_PREFIX:
				is_prefix_legal = self.prefix_pattern != "" and \
								  re.search(self.prefix_pattern, candidate_info[WORD_SUB_TREE],re.M) is not None
				is_prefix_illegal = self.illegal_prefix_pattern != "" and \
									re.search(self.illegal_prefix_pattern, candidate_info[WORD_SUB_TREE], re.M) is not None
			elif (candidate_position not in self.constant_positions and linked is None) or (linked is not None and candidate_position not in self.linked_positions[linked]):
				# The candidate is constant but it isn't suitable for this argument
				continue

			is_head_correct = self.check_head(candidate_info)

			if is_prefix_legal and not is_prefix_illegal and is_head_correct and self.check_possessive(candidate_info):
				return True

		return False

	def handle_linked_argument(self, dependency_tree, complement_type, extraction):

		for the_linked_arg in self.linked_positions.keys():
			candidates = get_argument_candidates(dependency_tree, extraction[the_linked_arg])

			for candidate_index in candidates:
				if self.check_constraints(dependency_tree, candidate_index, linked=the_linked_arg):
					extraction[complement_type] = candidate_index