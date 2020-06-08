import sys
import copy
import re
import os.path
import inflect
from collections import defaultdict
from nltk.corpus import wordnet as wn
inflect_engine = inflect.engine()

# Constants

det = "the"
should_print = True					# If false, then the program will print nothing (both to screen and to the output file)
should_print_to_screen = False		# If true, then the program will print to screen some debugging results
should_clean = True					# If true, then we do want a "clean" and updated results
should_replace_preps = False		# If true, then the preposition phrases will be replaced in the verbal process.
									# Oterwise, the comlex table will be updated programmatically.
shuffle_data = True					# If true, then the input data will be shuffled randomly
should_print_as_dataset = False		# If true, then the resulted output style will be like a dataset
									# Otherwise, the resulted output will be more readable

output_loc = sys.stdout
subcats_counts = {}
all_noms = {}
all_noms_backwards = {}

redundant_subcast = ["NOM-INTRANS", "NOM-INTRANS-RECIP"]



############################################# Dictionaries and Tables ############################################

def get_subentries_table():
	# (subentry, ud_links_list, [(how_to_find, limited_subcats, exception_subcats)])
	# ud_links_list is a list of lists\dicts of universal dependencies links (suitable for a subetry)
	# limited_subcats = [] means not limited
	# exception_subcats = [] means with no exceptions

	subentries_table = [
		("subject", 		[{"DET-POSS":[["poss"]], "N-N-MOD":[["compound"]], "PP-":[["prep_", "pobj"]]}],		[(["SUBJECT"], [], [])]),
		("ind-object", 		[{"DET-POSS":[["poss"]], "N-N-MOD":[["compound"]], "PP-":[["prep_", "pobj"]]}],		[(["PVAL1"], [], ["NOM-PP-FOR-TO-INF", "NOM-PP-TO-INF-RECIP", "NOM-PP-P-POSSING", "NOM-PP-WH-S", "NOM-PP-P-WH-S", "NOM-PP-HOW-TO-INF"]), (["IND-OBJ"], [], [])]),
		("object", 			[{"DET-POSS":[["poss"]], "N-N-MOD":[["compound"]], "PP-":[["prep_", "pobj"]]}],		[(["OBJECT"], [], ["NOM-NP-ING", "NOM-NP-ING-SC", "NOM-NP-ING-OC"])]),
		("pval", 			[["prep_", ["pobj"]]],																[(["PVAL"], [], ["NOM-P-NP-ING", "NOM-NP-P-NP-ING", "NOM-P-POSSING", "NOM-PP-P-POSSING"])]),
		("pval1", 			[["prep_", ["pobj"]]],																[(["PVAL1"], ["NOM-PP-FOR-TO-INF", "NOM-PP-TO-INF-RECIP", "NOM-PP-P-POSSING", "NOM-PP-WH-S", "NOM-PP-P-WH-S", "NOM-PP-HOW-TO-INF"], [])]),
		("pval2", 			[["prep_", ["pobj"]]],																[(["PVAL2"], [], [])]),
		("pval-nom", 		[],																					[(["PVAL-NOM"], [], [])]),
		("pval1-nom", 		[],																					[(["PVAL1-NOM"], [], [])]),
		("pval2-nom", 		[],																					[(["PVAL2-NOM"], [], [])]),
		("pval-ing", 		[["prep_", ["pcomp__ing"]]],														[(["NOM-SUBC", "P-ING", "PVAL"], [], ["NOM-ING-SC"])]), # P-ING
		("pval-poss-ing",	[["prep_", ["pcomp__ing"]], ["prep_", ["pobj__ing", ["poss"]]]],					[(["NOM-SUBC", "P-POSSING", "PVAL"], ["NOM-P-POSSING", "NOM-NP-P-POSSING"], ["NOM-POSSING", "NOM-POSSING-PP"]), (["PVAL"], ["NOM-P-POSSING", "NOM-PP-P-POSSING"], ["NOM-P-NP-ING", "NOM-NP-P-NP-ING", "NOM-POSSING-PP"])]), # P-POSSING
		("pval-comp-ing", 	[["prep_", ["pobj"], ["pcomp__ing"]]],												[(["PVAL"], ["NOM-P-NP-ING", "NOM-NP-P-NP-ING"], ["NOM-P-POSSING", "NOM-PP-P-POSSING", "NOM-POSSING-PP"])]), # P-NP-ING
		("pval-to-inf", 	[["advcl", ["mark_"], ["aux_to"]]],													[]), # P-TO-INF
		("pval-wh", 		[["prep_", ["pcomp", "mark_whether"]], ["prep_", ["pcomp", "dobj_what"]]],			[(["NOM-SUBC", "P-WH", "PVAL"], ["NOM-P-WH-S", "NOM-PP-P-WH-S", "NOM-NP-P-WH-S"], [])]), # P-WH
		("comp-ing", 		[["prep_", "pobj__ing", ["compound"]]],												[(["OBJECT"], ["NOM-NP-ING", "NOM-NP-ING-SC", "NOM-NP-ING-OC"], [])]), # NP-ING
		("ing", 			[["prep_", "pcomp__ing"]],															[(["NOM-SUBC", "P-ING", "PVAL"], ["NOM-ING-SC"], [])]), # just ING
		("poss-ing", 		[["prep_", "pcomp__ing"]],															[(["NOM-SUBC", "P-POSSING", "PVAL"], ["NOM-POSSING", "NOM-POSSING-PP"], ["NOM-P-POSSING", "NOM-NP-P-POSSING"])]), # just POSSING
		("adverb", 			[{"ADJP": [["amod"]], "ADVP": [["advmod"]]}],										[(["NOM-SUBC"], ["NOM-ADVP-PP", "NOM-NP-ADVP", "NOM-ADVP"], [])]),
		("sbar", 			[["acl", ["mark_that"]]], 															[]),
		("adjective", 		[["prep_", "amod"]], 																[]),
		("to-inf", 			[["acl", ["aux_to"]]],																[]), # TO-INF
		("wh",				[{"whether": [["prep_", "pcomp", ["mark_whether"]]],
							  "what": [["prep_", "pcomp", ["dobj_what"]]],
							  "how": [["prep_", "pcomp", ["advmod_how"]]]}],									[(["NOM-SUBC", "P-WH", "PVAL"], [], ["NOM-P-WH-S", "NOM-PP-P-WH-S", "NOM-NP-P-WH-S", "NOM-WHERE-WHEN-S", "NOM-PP-HOW-TO-INF"])]),
		("where-when",		[["prep_", "pcomp", ["advmod_where"]],
							 ["prep_", "pcomp", ["advmod_when"]],
							 ["prep_", "pcomp", ["dobj", "amod_much", "advmod_how"]],
							 ["prep_", "pcomp", ["dobj", "amod_many", "advmod_how"]]],							[(["NOM-SUBC", "P-WH", "PVAL"], ["NOM-WHERE-WHEN-S"], [])]), # just WHERE-WHEN (and how much and many)
		("how-to-inf",		[["prep_", "pcomp", ["advmod_how"]]],												[(["NOM-SUBC", "P-WH", "PVAL"], ["NOM-PP-HOW-TO-INF"], [])]) # HOW-TO-INF
	]

	return subentries_table

def a():
	# (subentry, ud_links_list, [(how_to_find, limited_subcats, exception_subcats)])
	# ud_links_list is a list of lists\dicts of universal dependencies links (suitable for a subetry)
	# limited_subcats = [] means not limited
	# exception_subcats = [] means with no exceptions

	subentries_table = [
		("subject",			[["nsubj"]], []),
		("object",			[["dobj"]], []),
		("ind-object",		[["dative"]], []),
		("adverb",			[["advmod"]], []),
		("pval",			[["prep"]], []),			# IF THERE ARE TWO PVALS THEN A SPLIT IS NEEDED TO PVAL1, PVAL2
		("pval-ing", 		[["prep", "pcomp__ing"]], []), # P-ING
		("pval-poss-ing", 	[["prep", "pcomp__ing", ["poss"]], ["pobj", "pcomp__ing", ["poss"]]], []), # P-ING
		("pval-comp-ing", 	[["prep", "pcomp__ing", ["nsubj"]]], []),  # P-ING
		("pval-to-inf", 	[["advcl", ["aux_to"], ["mark_for"]]], []),
		("pval-wh", 		[["prep", "pcomp", ["mark_whether"]], ["prep", "pcomp", ["dobj_what"]]], []), # P-ING
		("adjective",		[["prep_as", "amod"]], []),
		("sbar",			[["ccomp"]], []),
		("comp-ing",		[["ccomp__ing"]], []), # NP-ING
		("ing",				[["xcomp__ing"]], []), # just ING
		("poss-ing",		[["ccomp__ing", ["nsubj"]], ["ccomp__ing", ["poss"]]], []),
		("to-inf",			[["xcomp", ["aux_to"]]], []),
		("wh",				[["ccomp", ["mark_if"]], ["ccomp", ["mark_whether"]], ["ccomp", ["dobj_what"]], ["ccomp", ["advmod_how"]],
							 ["xcomp", ["mark_if"]], ["xcomp", ["mark_whether"]], ["xcomp", ["dobj_what"]], ["xcomp", ["advmod_how"]]], []),
		("how-to-inf",		[["xcomp", ["aux_to"], ["advmod_how"]]], []),
		("where-when",		[["ccomp", ["advmod_where"]], ["xcomp", ["advmod_where"]],
							 ["advcl", ["advmod_when"]], ["ccomp", ["advmod_when"]], ["xcomp", ["advmod_when"]],
							 ["ccomp", ["dobj", "amod_much", "advmod_how"]], ["ccomp", ["nsubj", "amod_much", "advmod_how"]], ["ccomp", ["dobj_much", "advmod_how"]],
							 ["ccomp", ["dobj", "amod_many", "advmod_how"]], ["ccomp", ["nsubj", "amod_many", "advmod_how"]], ["ccomp", ["dobj_many", "advmod_how"]]], [])
	]

	return subentries_table


def get_special_subcats_dict():
	# subcat: (required_list, list of (subentry, default_value) pairs)
	# About the required_list- should contain SUBJECT, OBJECT and order.
	# However, all the other subentries always required (according to the subcat structure).
	return {
		"NOM-NP":					(["OBJECT"],			[]),

		"NOM-ADVP":					([],					[("adverb", ["ADVP"])]),
		"NOM-ADVP-PP":				([],					[("adverb", ["ADVP"])]),
		"NOM-NP-ADVP":				(["OBJECT"],			[("adverb", ["ADVP"])]),

		"NOM-NP-TO-NP":				(["OBJECT"],			[("ind-object", ["PP-TO"])]),
		"NOM-NP-FOR-NP":			(["OBJECT"],			[("ind-object", ["PP-FOR"])]),

		"NOM-NP-AS-NP-SC":			(["OBJECT", "SUBJECT"],	[("pval", ["as"])]),
		"NOM-NP-AS-NP":				([],					[("pval", ["as"])]),
		"NOM-AS-NP":				([],					[("pval", ["as"])]),
		"NOM-NP-PP-AS-NP":			(["OBJECT", "order"],	[("pval2", ["as"])]),
		"NOM-NP-PP-PP":				(["OBJECT"],			[]),
		"NOM-NP-PP":				(["OBJECT"],			[]),

		"NOM-NP-AS-ADJP":			(["OBJECT"],			[("adjective", ["as"])]),

		"NOM-S":					([],					[("sbar", ["NOT NONE"])]),
		"NOM-THAT-S":				([],					[("sbar", ["NOT NONE"])]),
		"NOM-NP-S":					(["OBJECT"],			[("sbar", ["NOT NONE"])]),
		"NOM-PP-THAT-S":			([],					[("sbar", ["NOT NONE"])]),

		"NOM-NP-AS-ING":			(["OBJECT"],			[("pval-ing", ["as"])]),
		"NOM-NP-P-NP-ING":			(["OBJECT"],			[]),
		"NOM-NP-P-ING":				(["OBJECT"],			[]),
		"NOM-NP-P-ING-OC":			(["OBJECT"],			[]),
		"NOM-NP-P-ING-SC":			(["OBJECT"],			[]),
		"NOM-ING-SC":				([],					[("ing", "of")]),

		"NOM-NP-P-POSSING":			(["OBJECT"],			[]),
		"NOM-POSSING-PP":			([],					[("poss-ing", "of")]),
		"NOM-POSSING":				([],					[("poss-ing", "of")]),

		"NOM-PP-FOR-TO-INF":		([],					[("pval-to-inf", ["for"])]),
		"NOM-FOR-TO-INF":			([],					[("pval-to-inf", ["for"])]),
		"NOM-PP-TO-INF-RECIP":		([],					[("to-inf", ["to"])]),
		"NOM-P-NP-TO-INF":			([],					[("to-inf", ["to"])]),
		"NOM-P-NP-TO-INF-OC":		([],					[("to-inf", ["to"])]),
		"NOM-P-NP-TO-INF-VC":		([],					[("to-inf", ["to"])]),
		"NOM-NP-TO-INF-SC":			(["OBJECT"],			[("to-inf", ["to"])]),
		"NOM-NP-TO-INF-OC":			(["OBJECT"],			[("to-inf", ["to"])]),
		"NOM-NP-TO-INF-VC":			(["OBJECT"],			[("to-inf", ["to"])]),
		"NOM-TO-INF-SC":			(["OBJECT"],			[("to-inf", ["to"])]),

		"NOM-WH-S":					([],					[("wh", ["of"])]),
		"NOM-HOW-S":				([],					[("wh", ["of"])]),
		"NOM-NP-WH-S":				([],					[("wh", ["of"])]),
		"NOM-PP-WH-S":				([],					[("wh", ["of"])]),
		"NOM-WHERE-WHEN-S":			([],					[("where-when", ["of"])]),
		"NOM-PP-HOW-TO-INF":		([],					[("pval1", ["of"])])
	}

def replace_empty_list(a_list, replacement):
	"""
	Replacing the empty list somewhere in the list with the given replacement
	:param a_list: a list
	:param replacement: a replacement list
	:return: None
	"""

	temp = a_list
	last = a_list

	while temp != []:
		last = temp
		temp = temp[-1]

	last.pop()

	continue_list = replacement
	for x in continue_list:
		last.append(x)

def update_comlex_table(structure, tag, replacement):
	"""
	Updates the comlex table, by creating a similar structure to the given one
	The new structure replaces the given tag with the given replacement list
	:param structure: a given structure as a phrases list
	:param tag: a given string tag
	:param replacement: a replacemnt for the given tag
	:return: the resulted new structure, and a boolean that determines whether there was a replacement or not
	"""

	new_structure = []

	total_was_replaced = False

	for i in range(len(structure)):
		if type(structure[i]) == str:
			if tag == structure[i]:
				new_sub_structure = copy.deepcopy(replacement)

				if structure[i + 1:] != []:
					replace_empty_list(new_sub_structure, structure[i + 1:])

				new_structure.append(new_sub_structure)

				return new_structure, True

			new_structure.append(structure[i])
		else:
			new_sub_structure, was_replaced = update_comlex_table(structure[i], tag, replacement)

			if was_replaced:
				total_was_replaced = True

			new_structure.append(new_sub_structure)

	return new_structure, total_was_replaced

def get_comlex_table():
	# subcat, structure, suitable_pattern_entities
	# Be aware that the order matter, because the program try each line in that order and we want to find the most specific case
	comlex_table = [
		("NOM-WHERE-WHEN-S",			[[[["WRB_how", "JJ_many"]], "S"]],				["where-when"]),
		("NOM-WHERE-WHEN-S",			[[["WRB_how", "JJ_many"], "S"]],				["where-when"]),
		("NOM-WHERE-WHEN-S",			[[["WHADJP_how many"], "S"]],					["where-when"]),
		("NOM-WHERE-WHEN-S",			[[[["WRB_how", "JJ_much"]], "S"]],				["where-when"]),
		("NOM-WHERE-WHEN-S",			[[["WRB_how", "JJ_much"], "S"]],				["where-when"]),
		("NOM-WHERE-WHEN-S",			[[["WHADJP_how much"], "S"]],					["where-when"]),
		("NOM-WHERE-WHEN-S",			[[["WRB_where"], "S"]],							["where-when"]),
		("NOM-WHERE-WHEN-S",			[[["WRB_when"], "S"]],							["where-when"]),
		("NOM-PP-HOW-TO-INF",			[["WRB_how", [["TO_to"]]]],						["how-to-inf"]),
		("NOM-PP-HOW-TO-INF",			[[["WRB_how"], [["TO_to"]]]],					["how-to-inf"]),
		("NOM-PP-HOW-TO-INF",			[["WRB_how", "S"]],								["how-to-inf"]),
		("NOM-PP-HOW-TO-INF",			[[["WRB_how"], "S"]],							["how-to-inf"]),
		("NOM-PP-HOW-TO-INF",			["PP", ["WRB_how", [["TO_to"]]]],				["pval1", "how-to-inf"]),
		("NOM-PP-HOW-TO-INF",			["PP", [["WRB_how"], [["TO_to"]]]],				["pval1", "how-to-inf"]),
		("NOM-PP-HOW-TO-INF",			["PP", ["WRB_how", "S"]],						["pval1", "how-to-inf"]),
		("NOM-PP-HOW-TO-INF",			["PP", [["WRB_how"], "S"]],						["pval1", "how-to-inf"]),
		("NOM-NP-P-WH-S",				["NP", ["IN", ["IN_whether", "S"]]],			["object", "pval-wh"]),
		("NOM-NP-P-WH-S",				["NP", ["IN", [["WP_what"], "S"]]],				["object", "pval-wh"]),
		("NOM-NP-P-WH-S",				["NP", ["IN", [["WP_what"], "S"]]],				["object", "pval-wh"]),
		("NOM-PP-P-WH-S",				["PP", ["IN", ["IN_whether", "S"]]],			["pval1", "pval-wh"]),
		("NOM-PP-P-WH-S",				["PP", ["IN", ["WP_what", "S"]]],				["pval1", "pval-wh"]),
		("NOM-PP-P-WH-S",				["PP", ["IN", [["WP_what"], "S"]]],				["pval1", "pval-wh"]),
		("NOM-PP-WH-S",					["PP", ["IN_whether", "S"]],					["pval1", "wh"]),
		("NOM-PP-WH-S",					["PP", ["WP_what", "S"]],						["pval1", "wh"]),
		("NOM-PP-WH-S",					["PP", [["WP_what"], "S"]],						["pval1", "wh"]),
		("NOM-PP-WH-S",					["PP", ["IN_if", "S"]],							["pval1", "wh"]),
		("NOM-P-WH-S",					[["IN", ["IN_if", "S"]]],						["pval-wh"]),
		("NOM-P-WH-S",					[["IN", ["IN_whether", "S"]]],					["pval-wh"]),
		("NOM-P-WH-S",					[["IN", ["WP_what", "S"]]],						["pval-wh"]),
		("NOM-P-WH-S",					[["IN", [["WP_what"], "S"]]],					["pval-wh"]),
		("NOM-NP-WH-S",					["NP", ["IN_whether", "S"]],					["object", "wh"]),
		("NOM-HOW-S",					[["WRB_how", "S"]],								["wh"]),
		("NOM-HOW-S",					[[["WRB_how"], "S"]],							["wh"]),
		("NOM-WH-S",					[["WP_what", "S"]],								["wh"]),
		("NOM-WH-S",					[[["WP_what"], "S"]],							["wh"]),
		("NOM-WH-S",					[["IN_whether", "S"]],							["wh"]),
		("NOM-WH-S",					[["IN_if", "S"]],								["wh"]),

		# TO-INF- infinitival phrases
		("NOM-PP-FOR-TO-INF",			["PP", ["IN_for", ["NP", ["TO_to", ["VB"]]]]],	["pval1", "pval-to-inf"]),
		("NOM-FOR-TO-INF",				[["IN_for", ["NP", ["TO_to", ["VB"]]]]],		["pval-to-inf"]),
		("NOM-PP-TO-INF-RECIP",			["PP", [["TO_to", ["VB"]]]],					["pval1", "to-inf"]),
		("NOM-P-NP-TO-INF",				[["IN", "NP"], [["TO_to", ["VB"]]]],			["pval", "to-inf"]),
		("NOM-P-NP-TO-INF-OC",			[["IN", "NP"], [["TO_to", ["VB"]]]],			["pval", "to-inf"]),
		("NOM-P-NP-TO-INF-VC",			[["IN", "NP"], [["TO_to", ["VB"]]]],			["pval", "to-inf"]),
		("NOM-NP-TO-INF-VC",			["NP", [["TO_to", ["VB"]]]],					["object", "to-inf"]),
		("NOM-NP-TO-INF-VC",			[["NP", [["TO_to", ["VB"]]]]],					[["object", "to-inf"]]),
		("NOM-NP-TO-INF-SC",			["NP", [["TO_to", ["VB"]]]],					["object", "to-inf"]),
		("NOM-NP-TO-INF-SC",			[["NP", [["TO_to", ["VB"]]]]],					[["object", "to-inf"]]),
		("NOM-NP-TO-INF-OC",			["NP", [["TO_to", ["VB"]]]],					["object", "to-inf"]),
		("NOM-NP-TO-INF-OC",			[["NP", [["TO_to", ["VB"]]]]],					[["object", "to-inf"]]),
		("NOM-TO-INF-SC",				[[["TO_to", ["VB"]]]],							["to-inf"]),

		# POSSING- possesive gerunds
		("NOM-POSSING-PP",				[[["VBG", "PP"]]],								[[["poss-ing", "pval"]]]),
		("NOM-POSSING-PP",				[[["PRP$", "VBG", "NP", "PP"]]],				[[["poss-ing", "poss-ing", "poss-ing", "pval"]]]),
		("NOM-POSSING-PP",				[[[["PRP$"], "VBG", "NP", "PP"]]],				[[["poss-ing", "poss-ing", "poss-ing", "pval"]]]),
		("NOM-POSSING-PP",				[["PRP$", "VBG", "NP", "PP"]],					[["poss-ing", "poss-ing", "poss-ing", "pval"]]),
		("NOM-POSSING-PP",				[[["PRP$"], "VBG", "NP", "PP"]],				[["poss-ing", "poss-ing", "poss-ing", "pval"]]),
		("NOM-POSSING-PP",				[["NP__'s", ["VBG", "NP", "PP"]]],				[["poss-ing", ["poss-ing", "poss-ing", "pval"]]]),
		("NOM-POSSING-PP",				[["NP__s '", ["VBG", "NP", "PP"]]],				[["poss-ing", ["poss-ing", "poss-ing", "pval"]]]),
		("NOM-POSSING-PP",				["PP", ["VBG"]],								["pval", "poss-ing"]),
		("NOM-POSSING-PP",				["PP", [["VBG"]]],								["pval", "poss-ing"]),
		("NOM-POSSING-PP",				["PP", ["PRP$", ["VBG"]]],						["pval", "poss-ing"]),
		("NOM-POSSING-PP",				["PP", [["PRP$"], ["VBG"]]],					["pval", "poss-ing"]),
		("NOM-POSSING-PP",				["PP", ["NP__'s", ["VBG"]]],					["pval", "poss-ing"]),
		("NOM-POSSING-PP",				["PP", ["NP__s '", ["VBG"]]],					["pval", "poss-ing"]),
		("NOM-NP-P-POSSING",			["NP", ["IN", ["VBG"]]],						["object", "pval-poss-ing"]),
		("NOM-NP-P-POSSING",			["NP", ["IN", [["VBG"]]]],						["object", "pval-poss-ing"]),
		("NOM-NP-P-POSSING",			["NP", ["IN", [["PRP$"], ["VBG"]]]],			["object", "pval-poss-ing"]),
		("NOM-NP-P-POSSING",			["NP", ["IN", ["PRP$", ["VBG"]]]],				["object", "pval-poss-ing"]),
		("NOM-NP-P-POSSING",			["NP", ["IN", ["NP__'s", ["VBG"]]]],			["object", "pval-poss-ing"]),
		("NOM-NP-P-POSSING",			["NP", ["IN", ["NP__s '", ["VBG"]]]],			["object", "pval-poss-ing"]),
		("NOM-PP-P-POSSING",			["PP", ["IN", ["VBG"]]],						["pval1", "pval-poss-ing"]),
		("NOM-PP-P-POSSING",			["PP", ["IN", [["VBG"]]]],						["pval1", "pval-poss-ing"]),
		("NOM-PP-P-POSSING",			["PP", ["IN", [["PRP$"], ["VBG"]]]],			["pval1", "pval-poss-ing"]),
		("NOM-PP-P-POSSING",			["PP", ["IN", ["PRP$", ["VBG"]]]],				["pval1", "pval-poss-ing"]),
		("NOM-PP-P-POSSING",			["PP", ["IN", ["NP__'s", ["VBG"]]]],			["pval1", "pval-poss-ing"]),
		("NOM-PP-P-POSSING",			["PP", ["IN", ["NP__s '", ["VBG"]]]],			["pval1", "pval-poss-ing"]),
		("NOM-P-POSSING",				[["IN", ["VBG"]]],								["pval-poss-ing"]),
		("NOM-P-POSSING",				[["IN", [["VBG"]]]],							["pval-poss-ing"]),
		("NOM-P-POSSING",				[["IN", [["PRP$"], ["VBG"]]]],					["pval-poss-ing"]),
		("NOM-P-POSSING",				[["IN", ["PRP$", ["VBG"]]]],					["pval-poss-ing"]),
		("NOM-P-POSSING",				[["IN", ["NP__'s", ["VBG"]]]],					["pval-poss-ing"]),
		("NOM-P-POSSING",				[["IN", ["NP__s '", ["VBG"]]]],					["pval-poss-ing"]),
		("NOM-POSSING",					[[["VBG"]]],									["poss-ing"]),
		("NOM-POSSING",					[[["PRP$"], ["VBG"]]],							["poss-ing"]),
		("NOM-POSSING",					[[[["PRP$"], ["VBG"]]]],						["poss-ing"]),
		("NOM-POSSING",					[["PRP$", ["VBG"]]],							["poss-ing"]),
		("NOM-POSSING",					[[["PRP$", ["VBG"]]]],							["poss-ing"]),
		("NOM-POSSING",					[["NP__'s", ["VBG"]]],							["poss-ing"]),
		("NOM-POSSING",					[[["NP__'s", ["VBG"]]]],						["poss-ing"]),
		("NOM-POSSING",					[["NP__s '", ["VBG"]]],							["poss-ing"]),
		("NOM-POSSING",					[[["NP__s '", ["VBG"]]]],						["poss-ing"]),

		# ING- gerunds
		("NOM-NP-P-NP-ING",				["NP", ["IN", ["NP", ["VBG"]]]],				["object", "pval-comp-ing"]),
		("NOM-NP-P-NP-ING",				["NP", ["IN", ["NP", [["VBG"]]]]],				["object", "pval-comp-ing"]),
		("NOM-P-NP-ING",				[["IN", ["NP", ["VBG"]]]],						["pval-comp-ing"]),
		("NOM-P-NP-ING",				[["IN", ["NP", [["VBG"]]]]],					["pval-comp-ing"]),
		("NOM-NP-AS-ING",				["NP", ["IN_as", ["VBG"]]],						["object", "pval-ing"]),
		("NOM-NP-AS-ING",				["NP", ["IN_as", [["VBG"]]]],					["object", "pval-ing"]),
		("NOM-NP-P-ING",				["NP", ["IN", ["VBG"]]],						["object", "pval-ing"]),
		("NOM-NP-P-ING",				["NP", ["IN", [["VBG"]]]],						["object", "pval-ing"]),
		("NOM-NP-P-ING-OC",				["NP", ["IN", ["VBG"]]],						["object", "pval-ing"]),
		("NOM-NP-P-ING-OC",				["NP", ["IN", [["VBG"]]]],						["object", "pval-ing"]),
		("NOM-NP-P-ING-SC",				["NP", ["IN", ["VBG"]]],						["object", "pval-ing"]),
		("NOM-NP-P-ING-SC",				["NP", ["IN", [["VBG"]]]],						["object", "pval-ing"]),
		("NOM-P-ING-SC",				[["IN", ["VBG"]]],								["pval-ing"]),
		("NOM-P-ING-SC",				[["IN", [["VBG"]]]],							["pval-ing"]),
		("NOM-NP-ING",					[["NP", ["VBG"]]],								["comp-ing"]),
		("NOM-NP-ING",					[["NP", [["VBG"]]]],							["comp-ing"]),
		("NOM-NP-ING-OC",				[["NP", ["VBG"]]],								["comp-ing"]),
		("NOM-NP-ING-OC",				[["NP", [["VBG"]]]],							["comp-ing"]),
		("NOM-NP-ING-SC",				["NP", ["VBG"]],								"comp-ing"),
		("NOM-NP-ING-SC",				["NP", [["VBG"]]],								"comp-ing"),
		("NOM-ING-SC",					[["VBG"]],										["ing"]),
		("NOM-ING-SC",					[[["VBG"]]],									["ing"]),

		# SBAR
		("NOM-PP-THAT-S",				[["IN", "NP"], ["IN_that", "S"]],				[[None, "ind-object"], "sbar"]),
		("NOM-NP-S",					["NP", "S"],									["object", "sbar"]),				# Should be ["NP", ["S"]]- updated programmatically
		("NOM-NP-S",					["NP", ["IN_that", "S"]],						["object", "sbar"]),
		("NOM-THAT-S",					[["IN_that", "S"]],								["sbar"]),
		("NOM-S",						["S"],											["sbar"]),							# Should be [["S"]]- updated programmatically
		("NOM-S",						[["IN_that", "S"]],								["sbar"]),

		# Double pvals
		("NOM-NP-PP-AS-NP",				["NP", ["IN", "NP"], ["IN_as", "NP"]],			["object", [None, "ind-object"], "pval2"]),
		("NOM-NP-PP-AS-NP",				[["NP", ["IN", "NP"]], ["IN_as", "NP"]],		[["object", [None, "ind-object"]], "pval2"]),
		("NOM-NP-PP-PP",				["NP", "PP", "PP"],								["object", "pval", "pval2"]),
		("NOM-NP-PP-PP",				[["NP", "PP"], "PP"],							[["object", "pval"], "pval2"]),
		("NOM-PP-PP",					["PP", "PP"],									["pval", "pval2"]),

		# Both object and indirect-object
		("NOM-NP-TO-NP",				["NP", ["IN_to", "NP"]],						["object", [None, "ind-object"]]),
		("NOM-NP-TO-NP",				[["IN_to", "NP"], "NP"],						[[None, "ind-object"], "object"]),
		("NOM-NP-TO-NP",				["NP", "NP"],									["ind-object", "object"]),
		("NOM-NP-FOR-NP",				["NP", ["IN_for", "NP"]],						["object", [None, "ind-object"]]),
		("NOM-NP-FOR-NP",				[["IN_for", "NP"], "NP"],						[[None, "ind-object"], "object"]),
		("NOM-NP-FOR-NP",				["NP", "NP"],									["ind-object", "object"]),

		# Adjective
		("NOM-NP-AS-ADJP",				["NP", ["RB_as", "JJ"]],						["object", "adjective"]),
		("NOM-NP-AS-ADJP",				["NP", ["RB_as", "ADJP"]],						["object", "adjective"]),

		# Single pval
		("NOM-NP-AS-NP-SC",				["NP", ["IN_as", "NP"]],						["object", "pval"]),
		("NOM-NP-AS-NP",				["NP", ["IN_as", "NP"]],						["object", "pval"]),
		("NOM-AS-NP",					[["IN_as", "NP"]],								["pval"]),
		("NOM-NP-PP",					["NP", "PP"],									["object", "pval"]),

		# Double objects
		("NOM-NP-NP",					["NP", "NP"],									["ind-object", "object"]),

		# Adverb
		("NOM-ADVP-PP",					["ADVP", "PP"],									["adverb", "pval"]),
		("NOM-ADVP-PP",					["RB", "PP"],									["adverb", "pval"]),
		("NOM-NP-ADVP",					["NP", "ADVP"],									["object", "adverb"]),
		("NOM-NP-ADVP",					["NP", "RB"],									["object", "adverb"]),
		("NOM-ADVP",					["ADVP"],										["adverb"]),
		("NOM-ADVP",					["RB"],											["adverb"]),

		# Basic
		("NOM-PP",						["PP"],											["pval"]),
		("NOM-NP",						["NP"],											["object"])
	]

	special_preps_dict = get_special_preps_dict()
	updated_complex_table = []

	# Updating the comlex table programmatically
	if not should_replace_preps:
		for subcat, structure, suitable_pattern_entities in comlex_table:
			new_structures = []
			updated_complex_table.append((subcat, structure, suitable_pattern_entities))
			new_structures.append(update_comlex_table(structure, "S", ["S"]))

			# Creating new structures using the special preposition dictionary
			for prep, replacements in special_preps_dict.items():
				for replacement in replacements[2]:
					new_structure, was_replaced = update_comlex_table(structure, "IN", copy.deepcopy(replacement))

					if was_replaced:
						new_structures.append((new_structure, was_replaced))
						new_structures.append(update_comlex_table(new_structure, "S", ["S"]))

			for new_structure, was_replaced in new_structures:
				if was_replaced:
					updated_complex_table.append((subcat, new_structure, suitable_pattern_entities))
	else:
		updated_complex_table = comlex_table

	return updated_complex_table


def get_pronoun_dict():
	pronoun_dict = {
		"he":		["his", "him"],
		"she":		["her", "her"],
		"it":		["its", "it"],
		"they":		["their", "them"],
		"we":		["our", "us"],
		"i":		["my", "me"],
		"you":		["your", "you"]
	}

	return pronoun_dict

def get_special_preps_dict():
	"""
		This dictionary replace the prepositions with multi words appeared in the NOMLEX lexicos
		The founded multi-words prepositions in the lexicon are:
			'in favor of', 'in connection with', 'away from', 'with regard to', 'according to',
			'close to', 'in terms of', 'as to', 'inside of', 'next to', 'such as', 'due to', 'off of',
			'in regard to', 'ahead of', 'up to', 'out of', 'counter to', 'with respect to'
	"""

	# prep_name: (word_replacement, links_replacement, phrases_replacement)
	# word_replacement is a single preposition that can replace the special preposition (the meaning of the sentence may be changed)
	# links_replacement replaces prep_
	# phrases_Replaement replaces a phrase P\PP, meaning a phrase that start with IN
	# [] meanes other data the continue the role both in the dependency case and the phrases case
	special_preps_dict = {
		'in favor of': 				('toward',			(["prep_with", "pobj_favor", "prep_to", []],		["prep_with", ["pobj_favor", ["prep_to", []]]]),			[["IN_in", ["NP_favor", ["IN_of", []]]]]),
		'in connection with':		('over',			(["prep_in", "pobj_connection", "prep_with", []],	["prep_in", ["pobj_connection", ["prep_with", []]]]),		[["IN_in", ["NP_connection", ["IN_with", []]]]]),
		'away from': 				('at',				(["advmod_away", "prep_from", []],					["advmod_away", ["prep_from", []]]),						[[["RB_away"], ["IN_from", []]], ["RB_away", ["IN_from", []]]]),
		'with regard to': 			('with',			(["prep_with", "pobj_regard", "prep_to", []],		["prep_with", ["pobj_regard", ["prep_to", []]]]),			[["IN_with", ["NP_regard", ["IN_to", []]]]]),
		'according to': 			('of',				(["prep_according", "prep_to", []],					["prep_according", ["prep_to", []]]),						[["_according", ["IN_to", []]]]),
		'close to':					('near',			(["amod_close", "prep_to", []],						["amod_close", ["prep_to", []]]),							[["RB_close", ["IN_to", []]], [["RB_close"], ["IN_to", []]]]),
		'in terms of':				('concerning', 		(["prep_in", "pobj_terms", "prep_of", []],			["prep_in", ["pobj_terms", ["prep_of", []]]]),				[["IN_in", ["NP_terms", ["IN_of", []]]]]),
		'inside of':				('in',				(["advmod_inside", "prep_of", []],					["advmod_inside", ["prep_of", []]]),						[[["RB_inside"], ["IN_of", []]], ["RB_inside", ["IN_of", []]]]),
		'as to':					('about',			(["prep_as", "prep_to", []],						["prep_as", ["prep_to", []]]),								[["IN_as", ["IN_to", []]]]),
		'next to':					('beside',			(["advmod_next", "prep_to", []],					["advmod_next", ["prep_to", []]]),							[[["RB_next"], ["IN_to", []]], ["RB_next", ["IN_to", []]]]),
		'such as':					('like',			(["prep_as", ["amod_such"], []],					["prep_as", ["amod_such"], []]),							[["JJ_such", "IN_as", []]]),
		'due to':					('since',			(["amod_due", ["pcomp_to"], []],					["amod_due", ["pcomp_to"], []]),							[["_due", ["_to", []]]]),
		'off of':					('off',				(["prep_off", "prep_of", []],						["prep_off", ["prep_of", []]]),								[["IN_off", ["IN_of", []]]]),
		'in regard to':				('regarding',		(["prep_in", "pobj_regard", "prep_to", []],			["prep_in", ["pobj_regard", ["prep_to", []]]]),				[["IN_in", ["NP_regard", ["IN_to", []]]]]),
		'ahead of':					('before',			(["advmod_ahead", "prep_of", []],					["advmod_ahead", ["prep_of", []]]),							[[["RB_ahead"], ["IN_of"], []], ["RB_ahead", ["IN_of", []]]]),
		'up to':					('after',			(["prep_up", "prep_to", []],						["prep_up", ["prep_to", []]]),								[["IN_up", ["IN_to", []]]]),
		'out of':					('from',			(["prep_out", "prep_of", []],						["prep_out", ["prep_of", []]]),								[["IN_out", ["IN_of", []]]]),
		'counter to':				('against',			(["prep_counter", "prep_to", []],					["prep_counter", ["prep_to", []]]),							[[["RB_counter"], ["IN_to", []]], ["RB_counter", ["IN_to", []]]]),
		'with respect to':			('on',				(["prep_with", "pobj_respect", "prep_to", []],		["prep_with", ["pobj_respect", ["prep_to", []]]]),			[["IN_with", ["NP_respect", ["IN_to", []]]]])
	}

	return special_preps_dict


def build_catvar_dict(catvar_db_filename):
	"""
	Building the catvar dictionary using the catvar database file in the given location
	:param catvar_db_filename: the location of the catvar database file
	:return: the created catvar dictionary ({verb: nouns})
	"""

	catvar_db = {}
	count = 0

	# Moving over the catvar database file
	# Finding the verb and nouns in the same line, meaning in the same words family
	if os.path.exists(catvar_db_filename):
		with open(catvar_db_filename, "r") as catvar_db_file:
			for line in catvar_db_file.readlines():
				line = line.replace("\n", "").replace("\r", "")
				line = re.sub('\d', '', line)
				family_words = line.split("#")
				nouns = []
				verbs = []

				# Moving over the words in the current line
				for word in family_words:
					# Aggregating nouns of the family
					if word.endswith("_N%"):
						nouns.append(word.split("_")[0])

					# Aggregating verbs of the family
					elif word.endswith("_V%"):
						verbs.append(word.split("_")[0])

				# Ignoring poural nouns
				for noun in nouns:
					if noun + "es" in nouns:
						del nouns[nouns.index(noun + "es")]

					if noun + "s" in nouns:
						del nouns[nouns.index(noun + "s")]

				# Adding all the founded verbs with the founded nouns, in the current line
				for verb in verbs:
					count += len(nouns)
					catvar_db.update({verb: nouns})

	return catvar_db

subentries_table = get_subentries_table()
special_subcats_dict = get_special_subcats_dict()
comlex_table = get_comlex_table()
pronoun_dict = get_pronoun_dict()
special_preps_dict = get_special_preps_dict()

# Get all the possible subcats
temp_comlex_subcats = list(set([i[0] for i in comlex_table])) + ["NOM-INTRANS", "NOM-INTRANS-RECIP"]
comlex_subcats = []

# Clean the comlex subcats from redundant ones
for curr_subcat in temp_comlex_subcats:
	if curr_subcat not in redundant_subcast:
		comlex_subcats.append(curr_subcat)

catvar_dict = build_catvar_dict("catvar_Data/catvar21.signed")


################################################### Utilities ####################################################

def get_best_word(word, possible_list, preferable_endings):
	"""
	Returns the most relevant word in the possible list to the given word
	The most relevant is a word that starts the same as the given word
	Also prefer words according to the given list of preferable endings
	:param word: a word
	:param possible_list: a list of words
	:param preferable_endings: a list of strings of preferable endings of the wanted word
	:return: the most relevant word to the given word
	"""

	if possible_list == []:
		return None

	best_word = possible_list[0]
	best_subword_length = 0
	for possible_word in possible_list:
		i = 0
		while i < len(word) and i < len(possible_word) and possible_word[i] == word[i]:
			i += 1

		i -= 1
		if i > best_subword_length or (i == best_subword_length and any([possible_word.endswith(end) for end in preferable_endings])):
			best_subword_length = i
			best_word = possible_word

	return best_word

def get_adj(word):
	"""
	Returns the best adjective that relates to the given word (if no adjective was found, None is returned)
	:param word: a word
	:return: an adjective that is most relevant to the given word, or the given word (if no adjective was found)
	"""

	possible_adj = []
	for ss in wn.synsets(word):
		for lemmas in ss.lemmas():  # all possible lemmas
			for ps in lemmas.pertainyms():  # all possible pertainyms (the adjectives of a noun)
				possible_adj.append(ps.name())

	best_adj = get_best_word(word, possible_adj, ["able", "ible", "al", "ful", "ic", "ive", "less", "ous"])

	if best_adj:
		return best_adj

	return word

def get_adv(word):
	"""
	Returns the best adverb that relates to the given word (if no adverb was found, None is returned)
	:param word: a word
	:return: an adverb that is most relevant to the given word, or the given word (if no adverb was found)
	"""

	possible_adv = []
	for synset in list(wn.all_synsets('r')):
		if get_adj(synset.lemmas()[0].name()) == word:
			possible_adv.append(synset.lemmas()[0].name())

	best_adv = get_best_word(word, possible_adv, ["ly", "ward", "wise"])

	if best_adv:
		return best_adv

	return word


def arranged_print(input_to_print):
	if should_print:
		print(input_to_print, file=output_loc)
		output_loc.flush()

		# Printing also to screen if needed
		if output_loc != sys.stdout and should_print_to_screen:
			print(input_to_print)

def separate_line_print(input_to_print, indent_level=0):
	if should_print:
		indentation_str = ""
		for _ in range(indent_level):
			indentation_str += "--"

		if type(input_to_print) == list:
			for x in input_to_print:
				if type(x) == defaultdict:
					x = dict(x)

				arranged_print(str(indentation_str) + str(x))

		elif type(input_to_print) == dict:
			for tag, x in input_to_print.items():
				if x != []: # Print only if it is not an empty list (meaning only if it is worth printing)
					arranged_print(str(indentation_str) + str(tag) + ": ")
					separate_line_print(x, indent_level + 1)

def print_as_dataset(sentence, noms_arguments_list):
	if should_print:
		for nom, arguments_list in noms_arguments_list.items():
			nom_str = nom[0]
			for arguments in arguments_list:
				for argument, argument_value in arguments.items():
					arranged_print("\t".join([sentence, nom_str, argument, argument_value]))

def get_clean_nom(nom):
	return "".join([i for i in nom if not (i.isdigit() or i == "#")])

def get_all_of_noms(nomlex_entries):
	"""
	Returns a dictionary of all the nominalizations in the given nomlex entries
	:param nomlex_entries: a dictionary of nominalizations
	:return: dictionary of nominalizations ({nom: clean_nom}),
			 and a "backwards" dictionary of nominalizations ({clean_nom: [nom]})
	"""

	all_noms = {}
	all_noms_backwards = {}

	for nom in nomlex_entries.keys():
		clean_nom = get_clean_nom(nom)
		all_noms.update({nom: clean_nom})
		
		if clean_nom in all_noms_backwards.keys():
			all_noms_backwards.update({clean_nom: all_noms_backwards[clean_nom] + [nom]})
		else:
			all_noms_backwards.update({clean_nom: [nom]})

	return all_noms, all_noms_backwards