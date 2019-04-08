import itertools
from allennlp.predictors.constituency_parser import ConstituencyParserPredictor
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import inflect
inflect_engine = inflect.engine()

import DictsAndTables
from DictsAndTables import get_comlex_table, seperate_line_print, get_adj, get_all_of_noms, get_pronoun_dict, det
from ExtractNomlexPatterns import extract_nom_patterns
from NominalPatterns import clean_argument, get_dependency

should_print = DictsAndTables.should_print


################################################ Subcat Detection ###############################################

def get_nomlex_entries(entries, verb):
	"""
	Returns the relevant nominalization entries for a specific verb
	:param entries: a dictionary of all the entries in NOMLEX lexicon
	:param verb: the base verb
	:return: a dictionary that contain only the relevant entries for the given verb
	"""

	relevant_entries = {}

	for nom, entry in entries.items():
		if "VERB" in entry.keys() and entry["VERB"] == verb:
			relevant_entries.update({nom: entry})

	return relevant_entries

def process_phrases_tree(sent_phrases_tree, index):
	"""
	A recursive function that processes a phrases tree as string and returns the suitable dictionary representation of the same tree
	:param sent_phrases_tree: a phrases tree in string format
	:param index: the current index in the sentence
	:return: a dictionary that represents the given phrases tree
	"""

	sub_phrases_trees = []

	while sent_phrases_tree[index] != ")":
		if sent_phrases_tree[index] == "(":
			index, sub_phrases_tree = process_phrases_tree(sent_phrases_tree, index + 1)
			sub_phrases_trees.append(sub_phrases_tree)
		else:
			sub_phrases_trees.append(sent_phrases_tree[index])
			index += 1

	if len(sub_phrases_trees) == 2:
		new_phrase_tree = {sub_phrases_trees[0]: [sub_phrases_trees[1]]}
	else:
		new_phrase_tree = {sub_phrases_trees[0]: sub_phrases_trees[1:]}

	return index + 1, new_phrase_tree

def get_phrase(phrases_tree):
	"""
	Returns the phrase that the given phrase tree represents
	:param phrases_tree: a given phrases tree (dictionary)
	:return: a string value
	"""

	# String- a single word of a phrase (a leaf)
	if type(phrases_tree) == str:
		return phrases_tree

	str_phrase = ""

	# List- the subtrees of the phrases tree
	if type(phrases_tree) == list:
		for sub_phrases_tree in phrases_tree:
			if str_phrase != "":
				str_phrase += " "

			str_phrase += get_phrase(sub_phrases_tree)

		return str_phrase

	# Dictionary- a subtree of the phrases tree
	for _, sub_phrases_tree in phrases_tree.items():
		if type(sub_phrases_tree) == str:
			if str_phrase != "":
				str_phrase += " "

			str_phrase += sub_phrases_tree
		else:
			for sub_sub_phrases_tree in sub_phrases_tree:
				sub_sub_phrase = get_phrase(sub_sub_phrases_tree)

				if str_phrase != "":
					str_phrase += " "

				str_phrase += sub_sub_phrase

	return str_phrase

def search_phrase(phrases_tree, searched_tag):
	"""
	A recursive function that searches for a specific phrase in the given tree
	:param phrases_tree: the given tree that is being searched
	:param searched_tag: the wanted tag
	:return: a list of all the trees with the wanted phrase tag as root
	"""

	if type(phrases_tree) == str:
		return []

	wanted_phrases = []

	for phrase_tag, sub_phrase_tree in phrases_tree.items():
		if phrase_tag == searched_tag:
			wanted_phrases.append({phrase_tag: sub_phrase_tree})
		else:
			for sub_sub_phrase_tree in sub_phrase_tree:
				sub_wanted_phrases = search_phrase(sub_sub_phrase_tree, searched_tag)

				if sub_wanted_phrases != []:
					wanted_phrases += sub_wanted_phrases

	return wanted_phrases

def get_sub_phrases(phrases_tree, phrases_tags, arguments_types = None):
	"""
	This function tries to find the phrases tags in the given list as sub-phrases of the given phrases tree
	It also may match the founded phrases to the given argument types according to their order (only if
	:param phrases_tree: a dictionary that represents the phrases tree to search in
	:param phrases_tags: a phrases tags list, we are looking to find
	:param arguments_types: a list of types of arguments to match with the wanted phrases
	:return: the list of sub-phrases that were found (if all the sub-phrases tags were found) or []
			 In addition, the matching arguments dictionary is also returned (if the arguments types aren't given, None is returned)
	"""

	index = 0
	phrases = []

	arguments = defaultdict(str)

	if type(arguments_types) == str:
		phrases_tree = [{"S": phrases_tree}]
		phrases_tags = [phrases_tags]
		arguments_types = [arguments_types]

		return get_sub_phrases(phrases_tree, phrases_tags, arguments_types)

	# Moving over the sub-phrases list in the current phrases tree
	for sub_phrases_trees in phrases_tree:
		if type(sub_phrases_trees) != str:

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
											arguments[temp_argument] += " " + value
										else:
											arguments[temp_argument] = value
								else:
									arguments[arguments_types[index]] = get_phrase(sub_phrases_tree)
							phrases.append(inner_phrases)
							index += 1
						else:
							arguments = defaultdict(str)
							phrases = []

							if arguments_types:
								return phrases, arguments
							else:
								return phrases, None
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
							equal_values = get_phrase(sub_phrases_tree).endswith(value)

						if tag == temp_tag and equal_values:
							if arguments_types and arguments_types[index]:
								if arguments[arguments_types[index]] != "":
									arguments[arguments_types[index]] += " "
								arguments[arguments_types[index]] += get_phrase(sub_phrases_tree)
							phrases.append({phrases_tags[index]: sub_phrases_tree})
							index += 1
						else:
							arguments = defaultdict(str)
							phrases = []

							if arguments_types:
								return phrases, arguments
							else:
								return phrases, None

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

	predictor = ConstituencyParserPredictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")
	phrase_tree = predictor.predict(sent)['trees']

	# Moving over each line in the input file
	# Spacing up all the opening\closing brackets
	temp_splitted_line = phrase_tree.replace("(", " ( ").replace(")", " ) ").replace(") \n", ")\n").replace("\"", "").split(' ')
	splitted_line = []

	for i in range(len(temp_splitted_line)):
		if temp_splitted_line[i] != '':
			splitted_line.append(temp_splitted_line[i].replace('\n', ''))

	new_sent = splitted_line

	# Translates the phrases tree from string to dictionary
	_, phrases_tree = process_phrases_tree(new_sent, 1)

	default_arguments = defaultdict(str)
	possible_arguments = []

	# Use the first NP, VP pair that was found in the phrases tree
	if "S" not in phrases_tree.keys():
		return []

	np_vp_phrases_trees, _ = get_sub_phrases(phrases_tree["S"], ["NP", "VP"], ["NP", "VP"])

	if np_vp_phrases_trees != []:
		np_phrase_tree = np_vp_phrases_trees[0]
		vp_phrase_tree = np_vp_phrases_trees[1]

		wordnet_lemmatizer = WordNetLemmatizer()
		default_arguments["verb"] = wordnet_lemmatizer.lemmatize(get_phrase(vp_phrase_tree).split(" ")[0], 'v')
		default_arguments["subject"] = get_phrase(np_phrase_tree)

		complex_table = get_comlex_table()

		for subcat_info in complex_table:
			subcat, tags_phrases, suitable_arguments = subcat_info

			# Even if the suitable subcat was found, a general case may also work
			_, founded_arguments = get_sub_phrases(vp_phrase_tree["VP"][1:], tags_phrases, suitable_arguments)

			# Cleaning arguments
			for arg in founded_arguments.keys():
				founded_arguments[arg] = clean_argument(founded_arguments[arg])

			# Checking if a suitable subcat was found
			if list(founded_arguments.keys()) != []:
				default_arguments["subcat"] = subcat

				# Adding the updated arguments to the possible arguments list
				temp_arguments = default_arguments.copy()
				temp_arguments.update(founded_arguments)

				if "sbar" in temp_arguments.keys() and not temp_arguments["sbar"].lower().startswith("that "):
					temp_arguments["sbar"] = "that " + temp_arguments["sbar"]

				if "wh" in temp_arguments.keys():
					if not temp_arguments["wh"].lower().startswith("how to"):
						possible_arguments.append(temp_arguments)
				else:
					possible_arguments.append(temp_arguments)

		# NOM-INTRANS- always suitable subcat
		default_arguments["subcat"] = "NOM-INTRANS"
		possible_arguments.append(default_arguments.copy())

		# NOM-INTRANS-RECIP- always suitable subcat in case of plural subject NP
		if inflect_engine.singular_noun(get_phrase(np_phrase_tree)):
			default_arguments["subcat"] = "NOM-INTRANS-RECIP"
			possible_arguments.append(default_arguments.copy())

	return possible_arguments



################################################ Verbal to Nominal ###############################################

def process_a_sentence(sent):
	"""
	Processes a sentence, returns its relevant arguments
	:param sent: the sentence that was processed
	:return: all the possible founded arguments of the verb in the sentence (as a list of dictionaries)
	"""

	# Replacing the first upper letter only if the word isn't a name of something (using NER from spacy)
	dependency = get_dependency(sent)
	if dependency != [] and dependency[0][-2] == "":
		sent = sent[0].lower() + sent[1:]

	possible_arguments = detect_comlex_subcat(sent)

	seperate_line_print(possible_arguments)

	return possible_arguments

def build_pre_nom(pattern, arguments):
	"""
	Builds the pre nominalization sentence
	:param pattern: a dictionary of roles and their types
	:param arguments: an arguments dictionary, contains some arguments of a certain nominalization
	:return: the string before the nominalization (by default it must be in the next order- subject > indobject > object)
	"""

	# An argument appears in the pattern, but it wasn't found in the sentence (=> problem)
	if ("subject" not in arguments.keys() and "subject" in pattern.keys()) or \
			("object" not in arguments.keys() and "object" in pattern.keys()) or \
			("ind-object" not in arguments.keys() and "ind-object" in pattern.keys()):
		return ""

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
	:return: the cleaned sentence (or None = deleted)
	"""

	pronoun_dict = get_pronoun_dict()

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
	:param arguments: a dictionary of context arguments, that helps in the translation
	:return: list of suitable nominal sentences for the given data
	"""

	sentences = []

	if should_print: print(pattern)

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
				for pronoun, forms_list in get_pronoun_dict().items():
					if arguments[subentry] == pronoun or (pronoun[0].upper() + pronoun[1:]) == arguments[subentry]:
						arg = forms_list[1]

				post_preps.append([option.replace("PP-", "").lower(), arg])

			elif subentry in ["pval", "pval1", "pval2", "pval-ing", "pval-comp-ing", "adjective", "to-inf", "pval-to-inf", "pval-poss-ing", "pval-wh"]:
				splitted = arguments[subentry].split(" ")

				if option == "pval-nom" or option == "pval1-nom" or option == "pval2-nom":
					option = pattern[option]
					arguments[subentry] = option + " " + " ".join(splitted[1:])

				post_preps.append([option.lower(), " ".join(splitted[1:])])

			elif subentry == "sbar":
				if arguments[subentry].startswith("that"):
					post_preps.append([arguments[subentry]])
				else:
					post_preps.append(["that", arguments[subentry]])

			elif subentry in ["ing", "poss-ing", "where-when", "how-to-inf"]:
				post_preps.append([option.lower(), arguments[subentry]])

			elif subentry == "adverb" and option == "ADVP":
				post_preps.append([arguments["adverb"]])

			elif subentry == "wh":
				post_preps.append([option.lower(), " ".join(arguments[subentry].split(" ")[1:])])

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
	Assumption- the sentence contain only one verb
	:param nomlex_entries: the json formatted data of nomlex lexicon
	:param sent: a given verbal sentence
	:return: a list of nominal suitable sentences for the given sentence
	"""

	# Getting the arguments for the verb in the sentence (= processing the sentence)
	possible_arguments = process_a_sentence(sent)

	nom_sentences = []
	founded_noms = []

	# There may be many possible arguments dictionaries
	for arguments in possible_arguments:
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

	return list(set(nom_sentences))