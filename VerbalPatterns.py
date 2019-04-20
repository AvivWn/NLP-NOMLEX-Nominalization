import itertools
import re
from allennlp.predictors.constituency_parser import ConstituencyParserPredictor
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import inflect
inflect_engine = inflect.engine()
predictor = ConstituencyParserPredictor.from_path(
	"https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")

import DictsAndTables
from DictsAndTables import comlex_table, pronoun_dict, special_preps_dict, \
						   seperate_line_print, get_adj, get_all_of_noms, det
from ExtractNomlexPatterns import extract_nom_patterns
from NominalPatterns import get_dependency, clean_argument



# Constants

chars_count = 0



################################################ Subcat Detection ###############################################

def get_nomlex_entries(nomlex_entries, verb):
	"""
	Returns the relevant nominalization entries for a specific verb
	:param nomlex_entries: a dictionary of all the entries in NOMLEX lexicon
	:param verb: the base form verb
	:return: a dictionary that contains only the relevant entries for the given verb
	"""

	relevant_entries = {}

	for nom, entry in nomlex_entries.items():
		if "VERB" in entry.keys() and entry["VERB"] == verb:
			relevant_entries.update({nom: entry})

	return relevant_entries

def process_phrases_tree(sent_phrases_tree, index):
	"""
	A recursive function that processes a phrases tree as string and returns the suitable dictionary representation of the same tree
	:param sent_phrases_tree: a phrases tree in string format
	:param index: the current index in the sentence
	:return: a dictionary that represents the given phrases tree,
			 and an index (in order to know the last processed index in the string representation of the tree)
	"""
	global chars_count

	sub_phrases_trees = []

	while sent_phrases_tree[index] != ")":
		if sent_phrases_tree[index] == "(":
			sub_phrases_tree, index = process_phrases_tree(sent_phrases_tree, index + 1)
			sub_phrases_trees.append(sub_phrases_tree)
		else:
			sub_phrases_trees.append(sent_phrases_tree[index])
			index += 1

	for i  in range(len(sub_phrases_trees)):
		if i > 0 and type(sub_phrases_trees[i]) == str:
			sub_phrases_trees[i] = (sub_phrases_trees[i], chars_count, chars_count + len(sub_phrases_trees[i]) - 1)
			chars_count += len(sub_phrases_trees[i][0]) + 1

	if len(sub_phrases_trees) == 2:
		new_phrase_tree = {sub_phrases_trees[0]: [sub_phrases_trees[1]]}
	else:
		new_phrase_tree = {sub_phrases_trees[0]: sub_phrases_trees[1:]}

	return new_phrase_tree, index + 1

def get_phrase(phrases_tree):
	"""
	Returns the phrase that the given phrase tree represents
	:param phrases_tree: a given phrases tree (dictionary)
	:return: a string value, and start + end indexes in the sentence
	"""

	# tuple- a single word of a phrase (a leaf) with its place (index) in the sentence
	if type(phrases_tree) == tuple:
		return phrases_tree

	str_phrase = ""
	first_index = -1
	last_index = -1

	# List- the subtrees of the phrases tree
	if type(phrases_tree) == list:
		for sub_phrases_tree in phrases_tree:
			if str_phrase != "":
				str_phrase += " "

			curr_str_phrase, curr_first_index, curr_last_index = get_phrase(sub_phrases_tree)
			str_phrase += curr_str_phrase

			if curr_first_index < first_index or first_index == -1:
				first_index = curr_first_index

			if curr_last_index > last_index:
				last_index = curr_last_index

		return str_phrase, first_index, last_index

	# Dictionary- a subtree of the phrases tree
	if type(phrases_tree) == dict:
		for _, sub_phrases_tree in phrases_tree.items():
			if str_phrase != "":
				str_phrase += " "

			curr_str_phrase, curr_first_index, curr_last_index = get_phrase(sub_phrases_tree)
			str_phrase += curr_str_phrase

			if curr_first_index < first_index or first_index == -1:
				first_index = curr_first_index

			if curr_last_index > last_index:
				last_index = curr_last_index

	return str_phrase, first_index, last_index

def search_phrase(phrases_tree, searched_tag):
	"""
	A recursive function that searches for a specific phrase in the given tree
	:param phrases_tree: the given tree that is being searched
	:param searched_tag: the wanted tag
	:return: a list of all the sub-trees with the wanted phrase tag as root
	"""

	if type(phrases_tree) == str:
		return []

	wanted_subtrees = []

	for phrase_tag, sub_phrase_tree in phrases_tree.items():
		if phrase_tag == searched_tag:
			wanted_subtrees.append({phrase_tag: sub_phrase_tree})
		else:
			for sub_sub_phrase_tree in sub_phrase_tree:
				sub_wanted_phrases = search_phrase(sub_sub_phrase_tree, searched_tag)

				if sub_wanted_phrases != []:
					wanted_subtrees += sub_wanted_phrases

	return wanted_subtrees

def get_sub_phrases(phrases_tree, phrases_tags, arguments_types = None):
	"""
	This function tries to find the phrases tags in the given list as sub-phrases of the given phrases tree
	:param phrases_tree: a dictionary that represents the phrases tree to search in
	:param phrases_tags: a phrases tags list, we are looking to find
	:param arguments_types: a list of types of arguments to match with the wanted phrases (default is None)
	:return: the list of sub-phrases that were found (if all the sub-phrases tags were found) or []
			 In addition, the matching arguments dictionary is also returned (if the arguments types aren't given, None is returned)
	"""

	index = 0
	phrases = []

	default_arguments = defaultdict(list)
	default_phrases = []

	if not arguments_types:
		default_arguments = None

	arguments = defaultdict(list)

	if type(arguments_types) == str:
		phrases_tree = [{"S": phrases_tree}]
		phrases_tags = [phrases_tags]
		arguments_types = [arguments_types]

		return get_sub_phrases(phrases_tree, phrases_tags, arguments_types)

	# Moving over the sub-phrases list in the current phrases tree
	for sub_phrases_trees in phrases_tree:
		if type(sub_phrases_trees) != tuple:

			# Moving over the sub-phrases trees in the current sub-phrases list
			for tag, sub_phrases_tree in sub_phrases_trees.items():

				# Continue only if all the wanted phrases haven't found yet
				if index < len(phrases_tags) and (not arguments_types or index < len(arguments_types)):
					# Checking if this is a complex tag (a list)
					if type(phrases_tags[index]) == list:

						# Checking if the suitable argument is also complex of simple
						if arguments_types and type(arguments_types[index]) == list:
							inner_phrases, temp_arguments = get_sub_phrases(sub_phrases_tree, phrases_tags[index], arguments_types[index])
						else:
							inner_phrases, temp_arguments = get_sub_phrases(sub_phrases_tree, phrases_tags[index])

						if inner_phrases != []:
							if arguments_types:
								if temp_arguments:
									for temp_argument, value in temp_arguments.items():
										if temp_argument in arguments.keys():
											arguments[temp_argument] += [(" ", -1)] + value
										else:
											arguments[temp_argument] = [value]
								else:
									arguments[arguments_types[index]] = [get_phrase(sub_phrases_tree)]
							phrases.append(inner_phrases)
							index += 1
						else:
							return default_phrases, default_arguments
					else:
						# This is a simple tag (string)

						equal_values = True
						temp_tag = phrases_tags[index]

						# comparing the values in the right way
						splitted = temp_tag.split("_")
						if len(splitted) == 2:
							temp_tag, value = splitted
							equal_values = value == sub_phrases_tree[0]
						elif len(splitted) == 3:
							temp_tag, _, value = splitted
							equal_values = get_phrase(sub_phrases_tree)[0].endswith(value)

						if tag == temp_tag and equal_values:
							if arguments_types and arguments_types[index]:
								if arguments[arguments_types[index]] != []:
									arguments[arguments_types[index]] += [(" ", -1)]
								arguments[arguments_types[index]] += [get_phrase(sub_phrases_tree)]
							phrases.append({phrases_tags[index]: sub_phrases_tree})
							index += 1
						else:
							return default_phrases, default_arguments

				else:
					break

	if len(phrases_tags) != len(phrases):
		phrases = []
		arguments = defaultdict(str)

	if arguments_types:
		return phrases, arguments
	else:
		return phrases, None

def detect_comlex_subcat(sent):
	"""
	Detects the comblex sub-categorization of the given sentence
	:param sent: a sentence string
	:return: a list of arguments dictionaries with values that are relevant to each founded subcat
	"""
	global chars_count

	phrase_tree = predictor.predict_batch_json([{"sentence": sent}])[0]['trees']

	# Spacing up all the opening\closing brackets
	splitted_phrase_tree = phrase_tree.replace("(", " ( ").replace(")", " ) ").replace(") \n", ")\n").replace("\"", "").split(' ')
	new_splitted_phrase_tree = []

	for i in range(len(splitted_phrase_tree)):
		if splitted_phrase_tree[i] != '':
			new_splitted_phrase_tree.append(splitted_phrase_tree[i].replace('\n', ''))

	# Translates the phrases tree from string to dictionary
	chars_count = 0
	phrases_tree, _ = process_phrases_tree(new_splitted_phrase_tree, 1)

	default_arguments = defaultdict()
	possible_arguments = []

	# Use the first NP, VP pair that was found in the phrases tree
	if "S" not in phrases_tree.keys():
		return []

	np_vp_phrases_trees, _ = get_sub_phrases(phrases_tree["S"], ["NP", "VP"], ["NP", "VP"])

	if np_vp_phrases_trees != []:
		np_phrase_tree = np_vp_phrases_trees[0]
		vp_phrase_tree = np_vp_phrases_trees[1]

		wordnet_lemmatizer = WordNetLemmatizer()
		verb_phrase = get_phrase(vp_phrase_tree)
		default_arguments["verb"] = (wordnet_lemmatizer.lemmatize(verb_phrase[0].split(" ")[0], 'v'), verb_phrase[1], verb_phrase[2])
		default_arguments["subject"] = get_phrase(np_phrase_tree)

		for subcat_info in comlex_table:
			subcat, tags_phrases, suitable_arguments = subcat_info

			# Even if the suitable subcat was found, a general case may also work
			_, founded_arguments = get_sub_phrases(vp_phrase_tree["VP"][1:], tags_phrases, suitable_arguments)

			# Cleaning arguments
			for arg_str, arg_value in founded_arguments.items():
				first_index = arg_value[0][1]
				last_index = arg_value[-1][-1]
				arg_value = "".join([phrase for phrase, _, _ in arg_value])

				founded_arguments[arg_str] = (clean_argument(arg_value), first_index, last_index)

			# Checking if a suitable subcat was found
			if list(founded_arguments.keys()) != []:
				default_arguments["subcat"] = subcat

				# Adding the updated arguments to the possible arguments list
				curr_arguments = default_arguments.copy()
				curr_arguments.update(founded_arguments)

				if "wh" in curr_arguments.keys():
					if not curr_arguments["wh"][0].lower().startswith("how to"):
						possible_arguments.append(curr_arguments)
				else:
					possible_arguments.append(curr_arguments)

		# NOM-INTRANS- always suitable subcat
		default_arguments["subcat"] = "NOM-INTRANS"
		possible_arguments.append(default_arguments.copy())

		# NOM-INTRANS-RECIP- always suitable subcat in case of plural subject NP
		if inflect_engine.singular_noun(get_phrase(np_phrase_tree)[0]):
			default_arguments["subcat"] = "NOM-INTRANS-RECIP"
			possible_arguments.append(default_arguments.copy())

	return possible_arguments



################################################ Verbal to Nominal ###############################################

def extract_arguments_names(sent):
	"""
	Extracts the specific names of the arguments, and the new indexes of those arguments
	:param sent: a given sentence (string). The sentence may include the arguments names, like so:
				 [A0 IBM] appointed [A1 Alice Smith]
	:return: The new sentence (without the brackets),
			 and the arguments names that were found (list of tuples as [(arg_name, first_index, last_index)])
	"""

	arguments_names = {}
	last_arg = ("", 0)
	new_sent = []
	curr_index = 0
	in_arg = False

	# Getting the argument names (if they appear in the sentence), and their new indexes
	for word in sent.split(" "):
		if word.startswith("["):
			new_word = ""
			last_arg = (word[1:], curr_index)
			in_arg = True

		elif in_arg:
			if word.endswith("]"):
				new_word = word[:-1]
				word = new_word

				last_arg = (last_arg[0], last_arg[1])

				arguments_names[last_arg[0]] = (last_arg[1], curr_index + len(word) - 1)
				last_arg = ("", 0)

				in_arg = False
			else:
				new_word = word

				last_arg = (last_arg[0], last_arg[1])
		else:
			new_word = word

		if new_word != "":
			new_sent.append(new_word)
			curr_index += len(new_sent[-1]) + 1

	sent = " ".join(new_sent)

	if sent.endswith(" ."):
		sent = sent[:-2] + "."

	return sent, arguments_names

def process_sentence(sent):
	"""
	Processes a sentence, returns its relevant arguments
	:param sent: the sentence that should be processed. The sentence can contain specific arguments names
	:return: all the possible arguments of the verb in the sentence (as a list of dictionaries)
	"""

	# Cleaning the given sentence
	while sent.startswith(" "):
		sent = sent[1:]

	while sent.endswith(" "):
		sent = sent[:-1]

	sent = sent.replace("\n", "").replace("\r\n", "").replace("\r", "")

	if sent == "":
		return []

	if DictsAndTables.should_clean:
		# Replacing the first upper letter only if the word isn't a name of something (using NER from spacy)
		dependency = get_dependency(sent)
		if dependency != [] and dependency[0][-2] == "":
			sent = sent[0].lower() + sent[1:]

	# The sentence may contain a specific names of the arguments
	# We want to extract those names
	# Example [A0 IBM] appointed [A1 Alice Smith] => A0 = "IBM", A1 = "Alice Smith"
	sent, arguments_names = extract_arguments_names(sent)

	if DictsAndTables.should_print and DictsAndTables.should_print_to_screen:
		print("Arguments Names: " + str(arguments_names))
		print("Cleaned Sentence: " + sent)


	replaced_indexes = []

	# Replace the special prepositions only if needed
	if DictsAndTables.should_replace_preps:
		sent = " " + sent + " "
		temp_sent = sent.lower()

		# Replacing the special prepositions (prepositions that contain more than one word) with their replacemnet
		for original_prep, replace_prep in special_preps_dict.items():
			temp_sent = temp_sent.replace(" " + original_prep + " ", " " + replace_prep.upper() + " ")
			sent = sent.replace(" " + original_prep + " ", " " + replace_prep + " ")

		sent = sent[1:-1]
		temp_sent = temp_sent[1:-1]

		# Remember the replaced indexes
		for original_prep, replace_prep in special_preps_dict.items():
			replaced_indexes += [(m.start() + 1, original_prep, replace_prep) for m in re.finditer(" " + replace_prep.upper() + " ", temp_sent)]

		if DictsAndTables.should_print and DictsAndTables.should_print_to_screen:
			print("Sentence after replacing preps: " + sent)


	# Getting the possible arguments
	possible_arguments = detect_comlex_subcat(sent)

	possible_arguments_pairs = []
	new_possible_arguments = []

	# Fixing the arguments, according the founded arguments names, and the replaced indexes
	for arguments in possible_arguments:
		new_arguments = defaultdict()
		matching_arguments_names = {}

		for arg_str, arg_value in arguments.items():
			if type(arg_value) == tuple:
				arg_value, first_index, last_index = arg_value

				# Replace again the replaced indexes (Reventing to the original prepositions)
				for index, original_prep, replace_prep in replaced_indexes:
					if index == first_index:
						arg_value = original_prep + " " + " ".join(arg_value.split(" ")[1:])
					elif index >= first_index and index < first_index + arg_value:
						arg_value = arg_value[:index - first_index] + original_prep + arg_value[index - first_index + len(replace_prep):]

				# Updating the arguments names according to the founded names
				found = False
				for argument_name, indexes in arguments_names.items():
					arg_first_index, arg_last_index = indexes
					if arg_first_index == first_index and arg_last_index == last_index:
						new_arguments[argument_name] = arg_value
						matching_arguments_names[arg_str] = argument_name
						found = True

				if not found:
					new_arguments[arg_str] = arg_value

				arguments[arg_str] = arg_value
			else:
				arguments[arg_str] = arg_value
				new_arguments[arg_str] = arg_value

		possible_arguments_pairs.append((arguments, matching_arguments_names))
		new_possible_arguments.append(new_arguments)


	seperate_line_print(new_possible_arguments)
	#seperate_line_print(possible_arguments)

	return possible_arguments_pairs

def extract_args_from_verbal(nomlex_entries, sent):
	"""
	Finds the suitable arguments for each nominalization that was created from the main verb in the given sentence
	:param nomlex_entries: the entries of nomlex lexicon
	:param sent: a simple sentence with a main verb
	:return: a dictionary {nom: (arguments, matching_names)} where each nom was created from the main verb in the given sentence
	"""

	# Getting the patterns of arguments for the verb in the sentence (= processing the sentence)
	possible_arguments = process_sentence(sent)

	verb_arguments_for_noms = {}

	# There may be many possible arguments dictionaries
	for arguments, matching_names in possible_arguments:
		# Getting the relevant nominalization entries according to the verb that we found
		relevant_entries = get_nomlex_entries(nomlex_entries, arguments["verb"])

		# Extracting all the suitable nominalization patterns
		nom_patterns = extract_nom_patterns(relevant_entries, arguments["subcat"])

		for nominalization, patterns in nom_patterns.items():
			# The first suitable arguments list is preferable
			if nominalization not in verb_arguments_for_noms.keys():
				if patterns != []:
					verb_arguments_for_noms.update({nominalization: (arguments, matching_names)})
			else:
				break

	return verb_arguments_for_noms

def build_pre_nom(pattern, arguments):
	"""
	Builds the pre nominalization sentence (The sentence that appears before the nominalization)
	The arguments that can appear before the nominalization are: subject, indobject, object, comp-ing and adverb (ADJP only)
	:param pattern: a dictionary of subentries and their options
	:param arguments: a dictionary of arguments extracted for a certain nominalization
	:return: the string before the nominalization (by default it must be in the next order- subject > indobject > object)
	"""

	pre_nom = ""

	if "subject" in pattern.keys():
		if pattern["subject"] == "DET-POSS":
			pre_nom += arguments["subject"] + "'s "
		elif pattern["subject"] == "N-N-MOD":
			pre_nom += det + " " + arguments["subject"] + " "

	if "ind-object" in pattern.keys():
		if pattern["ind-object"] == "DET-POSS":
			pre_nom += arguments["ind-object"] + "'s "
		elif pattern["ind-object"] == "N-N-MOD":
			if pre_nom == "":
				pre_nom += det + " " + arguments["ind-object"] + " "
			else:
				pre_nom += arguments["ind-object"] + " "

	if "object" in pattern.keys():
		if pattern["object"] == "DET-POSS":
			pre_nom += arguments["object"] + "'s "
		elif pattern["object"] == "N-N-MOD":
			if pre_nom == "":
				pre_nom += det + " " + arguments["object"] + " "
			else:
				pre_nom += arguments["object"] + " "

	if "comp-ing" in pattern.keys():
		if pattern["comp-ing"] == "DET-POSS":
			pre_nom += arguments["comp-ing"] + "'s "
		elif pattern["comp-ing"] == "N-N-MOD":
			if pre_nom == "":
				pre_nom += det + " " + arguments["comp-ing"] + " "
			else:
				pre_nom += arguments["comp-ing"] + " "

	# Adding the adverb as adj if it is eval-adv (ADJP)
	if "adverb" in pattern.keys() and pattern["adverb"] == "ADJP":
		if pre_nom == "":
			pre_nom = det + " "

		adj = get_adj(arguments["adverb"])

		if adj:
			pre_nom += adj + " "

	if pre_nom == "":
		pre_nom = det + " "

	return pre_nom

def clean_sentence(sent):
	"""
	Cleans the sentence from mistakes with pronouns and more
	:param sent: a sentnece of words
	:return: the cleaned sentence, or None (= deleted)
	"""

	# Double determiners
	sent = sent.replace(" " + det + " the ", " the ").\
				replace(" " + det + " The ", " the ").\
		   		replace(" " + det + " A ", " a ").\
		   		replace(" " + det + " a ", " a ").\
		   		replace(" " + det + " an ", " an ").\
		   		replace(" " + det + " An ", " an ")

	# Pronouns
	# Translating the base form of the pronoun to the suitable form according to the context
	for pronoun, forms_list in pronoun_dict.items():
		# Pronouns and other forms + "'s"
		sent = sent.replace(" " + pronoun + "'s ", " " + forms_list[0] + " ").\
					replace(" " + pronoun[0].upper() + pronoun[1:] + "'s ", " " + forms_list[0] + " ")

		sent = sent.replace(" " + forms_list[1] + "'s ", " " + forms_list[0] + " "). \
					replace(" " + forms_list[1][0].upper() + forms_list[1][1:] + "'s ", " " + forms_list[0] + " ")

		# Pronouns + determiner
		sent = sent.replace(" " + det + " " + forms_list[0] + " ", " " + forms_list[0] + " "). \
					replace(" " + det + " " + forms_list[0].upper() + forms_list[0][1:] + " ", " " + forms_list[0] + " ")

		if " " + det + " " + pronoun + " " in sent or " " + det + " " + pronoun[0].upper() + pronoun[1:] + " " in sent:
			return None

	sent = sent.replace(" i ", " I ")

	return sent

def pattern_to_sent(nominalization, pattern, arguments):
	"""
	Translates a single pattern into a sentence\s, using context arguments
	:param nominalization: the nominalization word
	:param pattern: a pattern, that should be translated (as a dictionary)
	:param arguments: a dictionary of context arguments, that used in the translation
	:return: list of suitable nominal sentences for the given input
	"""

	sentences = []

	# An argument appears in the pattern, but it wasn't found in the sentence (=> problem)
	# It shouldn't happen
	if any(x not in arguments.keys() for x in pattern.keys()):
		assert "problem with pattern: " + str(dict(pattern)) + " and the arguments: " + str(dict(arguments))

	if DictsAndTables.should_print and DictsAndTables.should_print_to_screen:
		print(dict(pattern))

	# Getting the sentence string before the nominalization
	pre_nom = build_pre_nom(pattern, arguments)

	# Adding the nominalization
	sentence = pre_nom + nominalization

	# Getting all the prepositions which appear in the pattern
	# Those prepositions will appear after the nominalization in any order
	post_preps = []
	for subentry, option in pattern.items():
		if subentry in arguments.keys():
			if type(option) == str and option.startswith("PP-"):

				# Fixing msitakes of pronouns after prepositions (only for the resulted sentence)
				arg = arguments[subentry]
				for pronoun, forms_list in pronoun_dict.items():
					if arguments[subentry] == pronoun or (pronoun[0].upper() + pronoun[1:]) == arguments[subentry]:
						arg = forms_list[1]

				post_preps.append([option.replace("PP-", "").lower(), arg])

			elif subentry in ["pval", "pval1", "pval2", "pval-ing", "pval-comp-ing", "adjective", "to-inf", "pval-to-inf", "pval-poss-ing", "pval-wh", "wh"]:
				splitted = arguments[subentry].split(" ")

				starter_prep_length = 1
				for prep in special_preps_dict.keys():
					if arguments[subentry].startswith(prep):
						starter_prep_length = len(prep.split(" "))

				if option == "pval-nom" or option == "pval1-nom" or option == "pval2-nom":
					option = pattern[option]
					arguments[subentry] = option + " " + " ".join(splitted[starter_prep_length:])

				post_preps.append([option.lower(), " ".join(splitted[starter_prep_length:])])

			elif subentry == "sbar":
				if arguments[subentry].startswith("that"):
					post_preps.append([arguments[subentry]])
				else:
					post_preps.append(["that", arguments[subentry]])

			elif subentry in ["ing", "poss-ing", "where-when", "how-to-inf"]:
				post_preps.append([option.lower(), arguments[subentry]])

			elif subentry == "adverb" and option == "ADVP":
				post_preps.append([arguments["adverb"]])

	# Finally, adding the relevant prepositions from the pattern (in any order)
	for preps_order in itertools.permutations(post_preps, len(post_preps)):
		temp_sentence = sentence
		for prep in preps_order:
			if len(prep) == 1:
				temp_sentence += " " + prep[0]
			else:
				temp_sentence += " " + prep[0] + " " + prep[1]

		sentences.append(temp_sentence)

	# Cleaning the resulted sentences
	new_sentences = []
	for i in range(len(sentences)):
		sentences[i] = " " + sentences[i] + " "
		temp = clean_sentence(sentences[i])

		if temp:
			sentences[i] = temp
			sentences[i] = sentences[i][1:-1] # Without the added spaces
			sentences[i] = sentences[i][0].upper() + sentences[i][1:] + "."
			new_sentences.append(sentences[i])

	return new_sentences

def verbal_to_nominal(nomlex_entries, sent):
	"""
	Translates a verbal sentence into a nominal sentence, using nominalizations
	Assumption- the sentence contains only one verb, the function refers only to the first verb
	:param nomlex_entries: the json formatted data of nomlex lexicon
	:param sent: a given verbal sentence
	:return: a list of nominal suitable sentences for the given sentence
	"""

	# Getting the patterns of arguments for the verb in the sentence (= processing the sentence)
	possible_arguments = process_sentence(sent)

	nom_sentences = []
	founded_noms = []

	# There may be many possible arguments dictionaries
	for arguments, _ in possible_arguments:
		# Getting the relevant nominalization entries according to the verb that we found
		relevant_entries = get_nomlex_entries(nomlex_entries, arguments["verb"])

		# Extracting all the suitable nominalization patterns
		nom_patterns = extract_nom_patterns(relevant_entries, arguments["subcat"])

		# Creating all the nominalization suitable sentences for the given sentence
		for nominalization, patterns in nom_patterns.items():
			if nominalization not in founded_noms:
				for pattern in patterns:
					nom_sentences += pattern_to_sent(get_all_of_noms(nomlex_entries)[nominalization], pattern, arguments)
					founded_noms.append(nominalization) # The first suitable arguments list is preferable
			else:
				break

	return list(set(nom_sentences))