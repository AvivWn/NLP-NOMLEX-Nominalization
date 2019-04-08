from nltk.corpus import wordnet as wn
import spacy
import inflect

inflect_engine = inflect.engine()
nlp = spacy.load('en_core_web_sm')

# Constants

det = "the"
should_print = True

############################################# Dictionaries and Tables ############################################

def get_subentries_table():
	# (subentry, ud_links_list, [(how_to_find, limited_subcats, exception_subcats)])
	# ud_links_list is a list of lists\dicts of universal dependencies links (suitable for a subetry)
	# limited_subcats = [] means not limited
	# exception_subcats = [] means with no exceptions
	return [
		("subject", 		[{"DET-POSS":["poss"], "N-N-MOD":["compound"], "PP-":["prep_", "pobj"]}],	[(["SUBJECT"], [], [])]),
		("ind-object", 		[{"DET-POSS":["poss"], "N-N-MOD":["compound"], "PP-":["prep_", "pobj"]}],	[(["PVAL1"], [], ["NOM-PP-FOR-TO-INF", "NOM-PP-TO-INF-RECIP", "NOM-PP-P-POSSING", "NOM-PP-WH-S", "NOM-PP-P-WH-S", "NOM-PP-HOW-TO-INF"]), ("IND-OBJ", [], [])]),
		("object", 			[{"DET-POSS":["poss"], "N-N-MOD":["compound"], "PP-":["prep_", "pobj"]}],	[(["OBJECT"], [], ["NOM-NP-ING", "NOM-NP-ING-SC", "NOM-NP-ING-OC"])]),
		("pval", 			[["prep_", ["pobj"]]],														[(["PVAL"], [], ["NOM-P-NP-ING", "NOM-NP-P-NP-ING", "NOM-P-POSSING", "NOM-PP-P-POSSING"])]),
		("pval1", 			[["prep_", ["pobj"]]],														[(["PVAL1"], ["NOM-PP-FOR-TO-INF", "NOM-PP-TO-INF-RECIP", "NOM-PP-P-POSSING", "NOM-PP-WH-S", "NOM-PP-P-WH-S", "NOM-PP-HOW-TO-INF"], [])]),
		("pval2", 			[["prep_", ["pobj"]]],														[(["PVAL2"], [], [])]),
		("pval-nom", 		[],																			[(["PVAL-NOM"], [], [])]),
		("pval1-nom", 		[],																			[(["PVAL1-NOM"], [], [])]),
		("pval2-nom", 		[],																			[(["PVAL2-NOM"], [], [])]),
		("pval-ing", 		[["prep_", ["pcomp__ing"]]],												[(["NOM-SUBC", "P-ING", "PVAL"], [], ["NOM-ING-SC"])]), # P-ING
		("pval-poss-ing",	[["prep_", ["pcomp__ing"]]],												[(["NOM-SUBC", "P-POSSING", "PVAL"], ["NOM-P-POSSING", "NOM-NP-P-POSSING"], ["NOM-POSSING", "NOM-POSSING-PP"]), (["PVAL"], ["NOM-P-POSSING", "NOM-PP-P-POSSING"], ["NOM-P-NP-ING", "NOM-NP-P-NP-ING", "NOM-POSSING-PP"])]), # P-POSSING
		("pval-comp-ing", 	[["prep_", ["pobj"], ["pcomp__ing"]]],										[(["PVAL"], ["NOM-P-NP-ING", "NOM-NP-P-NP-ING"], ["NOM-P-POSSING", "NOM-PP-P-POSSING", "NOM-POSSING-PP"])]), # P-NP-ING
		("pval-to-inf", 	[["advcl", ["mark_"], ["aux_to"]]],											[]), # P-TO-INF
		("pval-wh", 		[["prep_", ["pcomp", "mark_whether"]], ["prep_", ["pcomp", "dobj_what"]]],	[(["NOM-SUBC", "P-WH", "PVAL"], ["NOM-P-WH-S", "NOM-PP-P-WH-S", "NOM-NP-P-WH-S"], [])]), # P-WH
		("comp-ing", 		[["prep_", "pobj__ing"]],													[(["OBJECT"], ["NOM-NP-ING", "NOM-NP-ING-SC", "NOM-NP-ING-OC"], [])]), # NP-ING
		("ing", 			[["prep_", "pcomp__ing"]],													[(["NOM-SUBC", "P-ING", "PVAL"], ["NOM-ING-SC"], [])]), # just ING
		("poss-ing", 		[["prep_", "pcomp__ing"]],													[(["NOM-SUBC", "P-POSSING", "PVAL"], ["NOM-POSSING", "NOM-POSSING-PP"], ["NOM-P-POSSING", "NOM-NP-P-POSSING"])]), # just POSSING
		("adverb", 			[{"ADJP": ["amod"], "ADVP": ["advmod"]}],									[(["NOM-SUBC"], ["NOM-ADVP-PP", "NOM-NP-ADVP", "NOM-ADVP"], [])]),
		("sbar", 			[["acl", ["mark_that"]]], 													[]),
		("adjective", 		[["prep_", "amod"]], 														[]),
		("to-inf", 			[["acl", ["aux_to"]]],														[]), # TO-INF
		("wh",				[{"whether": ["prep_", "pcomp", ["mark_whether"]], "what": ["prep_", "pcomp", ["dobj_what"]], "how": ["prep_", "pcomp", ["advmod_how"]]}],	[(["NOM-SUBC", "P-WH", "PVAL"], [], ["NOM-P-WH-S", "NOM-PP-P-WH-S", "NOM-NP-P-WH-S", "NOM-WHERE-WHEN-S", "NOM-PP-HOW-TO-INF"])]),
		("where-when",		[["prep_", "pcomp", ["advmod_where"]], ["prep_", "pcomp", ["advmod_when"]], ["prep_", "pcomp", ["dobj", "amod_much", "advmod_how"]], ["prep_", "pcomp", ["dobj", "amod_many", "advmod_how"]]],[(["NOM-SUBC", "P-WH", "PVAL"], ["NOM-WHERE-WHEN-S"], [])]), # just WHERE-WHEN (and how much and many)
		("how-to-inf",		[["prep_", "pcomp", ["advmod_how"]]],										[(["NOM-SUBC", "P-WH", "PVAL"], ["NOM-PP-HOW-TO-INF"], [])]) # HOW-TO-INF
	]

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

		"NOM-WH-S":					([],					[("wh", ["of"])]),
		"NOM-HOW-S":				([],					[("wh", ["of"])]),
		"NOM-NP-WH-S":				([],					[("wh", ["of"])]),
		"NOM-PP-WH-S":				([],					[("wh", ["of"])]),
		"NOM-WHERE-WHEN-S":			([],					[("where-when", ["of"])]),
		"NOM-PP-HOW-TO-INF":		([],					[("pval1", ["of"])])
	}

def get_comlex_table():
	# subcat, structure, suitable_pattern_entities
	# Be aware that the order matter, because the program try each line in that order and we want to find the most specific case
	comlex_table = [
		 ("NOM-PP-HOW-TO-INF",			[["WRB_how", [["TO_to"]]]],						["how-to-inf"]),
		 ("NOM-PP-HOW-TO-INF",			[[["WRB_how"], [["TO_to"]]]],					["how-to-inf"]),
		 ("NOM-PP-HOW-TO-INF",			[["WRB_how", "S"]],								["how-to-inf"]),
		 ("NOM-PP-HOW-TO-INF",			[[["WRB_how"], "S"]],							["how-to-inf"]),
		 ("NOM-PP-HOW-TO-INF",			["PP", ["WRB_how", [["TO_to"]]]],				["pval1", "how-to-inf"]),
		 ("NOM-PP-HOW-TO-INF",			["PP", [["WRB_how"], [["TO_to"]]]],				["pval1", "how-to-inf"]),
		 ("NOM-PP-HOW-TO-INF",			["PP", ["WRB_how", "S"]],						["pval1", "how-to-inf"]),
		 ("NOM-PP-HOW-TO-INF",			["PP", [["WRB_how"], "S"]],						["pval1", "how-to-inf"]),
		 ("NOM-WHERE-WHEN-S",			[[[["WRB_how", "JJ_many"]], "S"]],				["where-when"]),
		 ("NOM-WHERE-WHEN-S",			[[["WRB_how", "JJ_many"], "S"]],				["where-when"]),
		 ("NOM-WHERE-WHEN-S",			[[[["WRB_how", "JJ_much"]], "S"]],				["where-when"]),
		 ("NOM-WHERE-WHEN-S",			[[["WRB_how", "JJ_much"], "S"]],				["where-when"]),
		 ("NOM-WHERE-WHEN-S",			[[["WRB_where"], "S"]],							["where-when"]),
		 ("NOM-WHERE-WHEN-S",			[[["WRB_when"], "S"]],							["where-when"]),
		 ("NOM-NP-P-WH-S",				["NP", ["IN", ["IN_whether", "S"]]],			["object", "pval-wh"]),
		 ("NOM-NP-P-WH-S",				["NP", ["IN", [["WP_what"], "S"]]],				["object", "pval-wh"]),
		 ("NOM-NP-P-WH-S",				["NP", ["IN", [["WP_what"], "S"]]],				["object", "pval-wh"]),
		 ("NOM-PP-P-WH-S",				["PP", ["IN", ["IN_whether", "S"]]],			["pval1", "pval-wh"]),
		 ("NOM-PP-P-WH-S",				["PP", ["IN", ["WP_what", "S"]]],				["pval1", "pval-wh"]),
		 ("NOM-PP-P-WH-S",				["PP", ["IN", [["WP_what"], "S"]]],				["pval1", "pval-wh"]),
		 ("NOM-PP-WH-S",				["PP", ["IN_whether", "S"]],					["pval1", "wh"]),
		 ("NOM-PP-WH-S",				["PP", ["WP_what", "S"]],						["pval1", "wh"]),
		 ("NOM-PP-WH-S",				["PP", [["WP_what"], "S"]],						["pval1", "wh"]),
		 ("NOM-PP-WH-S",				["PP", ["IN_if", "S"]],							["pval1", "wh"]),
		 ("NOM-P-WH-S",					[["IN", ["IN_if", "S"]]],						["pval-wh"]),
		 ("NOM-P-WH-S",					[["IN", ["IN_whether", "S"]]],					["pval-wh"]),
		 ("NOM-P-WH-S",					[["IN", ["WP_what", "S"]]],						["pval-wh"]),
		 ("NOM-P-WH-S",					[["IN", [["WP_what"], "S"]]],					["pval-wh"]),
		 ("NOM-NP-WH-S",				["NP", ["IN_whether", "S"]],					["object", "wh"]),
		 ("NOM-HOW-S",					[["WRB_how", "S"]],								["wh"]),
		 ("NOM-HOW-S",					[[["WRB_how"], "S"]],							["wh"]),
		 ("NOM-WH-S",					[["WP_what", "S"]],								["wh"]),
		 ("NOM-WH-S",					[[["WP_what"], "S"]],							["wh"]),
		 ("NOM-WH-S",					[["IN_whether", "S"]],							["wh"]),
		 ("NOM-WH-S",					[["IN_if", "S"]],								["wh"]),

		 # TO-INF- infinitival phrases
		 ("NOM-PP-FOR-TO-INF",			["PP", ["IN_for", ["NP", ["TO_to", ["VB"]]]]],	["pval1", "pval-to-inf"]),
		 ("NOM-FOR-TO-INF",				[["IN_for", ["NP", ["TO_to", ["VB"]]]]],		["pval-to-inf"]),
		 ("NOM-PP-TO-INF-RECIP",		["PP", [["TO_to", ["VB"]]]],					["pval1", "to-inf"]),
		 ("NOM-P-NP-TO-INF",			[["IN", "NP"], [["TO_to", ["VB"]]]],			["pval", "to-inf"]),
		 ("NOM-P-NP-TO-INF-OC",			[["IN", "NP"], [["TO_to", ["VB"]]]],			["pval", "to-inf"]),
		 ("NOM-P-NP-TO-INF-VC",			[["IN", "NP"], [["TO_to", ["VB"]]]],			["pval", "to-inf"]),
		 ("NOM-NP-TO-INF-VC",			["NP", [["TO_to", ["VB"]]]],					["object", "to-inf"]),
		 ("NOM-NP-TO-INF-SC",			["NP", [["TO_to", ["VB"]]]],					["object", "to-inf"]),
		 ("NOM-NP-TO-INF-OC",			["NP", [["TO_to", ["VB"]]]],					["object", "to-inf"]),
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
		 ("NOM-POSSING",				[[["VBG"]]],									["poss-ing"]),
		 ("NOM-POSSING",				[[["PRP$"], ["VBG"]]],							["poss-ing"]),
		 ("NOM-POSSING",				[["PRP$", ["VBG"]]],							["poss-ing"]),
		 ("NOM-POSSING",				[["NP__'s", ["VBG"]]],							["poss-ing"]),
		 ("NOM-POSSING",				[["NP__s '", ["VBG"]]],							["poss-ing"]),

		 # ING- gerunds
		 ("NOM-NP-P-NP-ING",			["NP", ["IN", ["NP", ["VBG"]]]],				["object", "pval-comp-ing"]),
		 ("NOM-NP-P-NP-ING",			["NP", ["IN", ["NP", [["VBG"]]]]],				["object", "pval-comp-ing"]),
		 ("NOM-P-NP-ING",				[["IN", ["NP", ["VBG"]]]],						["pval-comp-ing"]),
		 ("NOM-P-NP-ING",				[["IN", ["NP", [["VBG"]]]]],					["pval-comp-ing"]),
		 ("NOM-NP-AS-ING",				["NP", ["IN_as", ["VBG"]]],						["object", "pval-ing"]),
		 ("NOM-NP-AS-ING",				["NP", ["IN_as", [["VBG"]]]],					["object", "pval-ing"]),
		 ("NOM-NP-P-ING",				["NP", ["IN", ["VBG"]]],						["object", "pval-ing"]),
		 ("NOM-NP-P-ING",				["NP", ["IN", [["VBG"]]]],						["object", "pval-ing"]),
		 ("NOM-NP-P-ING-OC",			["NP", ["IN", ["VBG"]]],						["object", "pval-ing"]),
		 ("NOM-NP-P-ING-OC",			["NP", ["IN", [["VBG"]]]],						["object", "pval-ing"]),
		 ("NOM-NP-P-ING-SC",			["NP", ["IN", ["VBG"]]],						["object", "pval-ing"]),
		 ("NOM-NP-P-ING-SC",			["NP", ["IN", [["VBG"]]]],						["object", "pval-ing"]),
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
		 ("NOM-NP-S",					["NP", ["S"]],									["object", "sbar"]),
		 ("NOM-NP-S",					["NP", ["IN_that", "S"]],						["object", "sbar"]),
		 ("NOM-THAT-S",					[["IN_that", "S"]],								["sbar"]),
		 ("NOM-S",						[["S"]],										["sbar"]),
		 ("NOM-S",						[["IN_that", "S"]],								["sbar"]),

		 # Double pvals
		 ("NOM-NP-PP-AS-NP",			["NP", ["IN", "NP"], ["IN_as", "NP"]],			["object", [None, "ind-object"], "pval2"]),
		 ("NOM-NP-PP-AS-NP",			[["NP", ["IN", "NP"]], ["IN_as", "NP"]],		[["object", [None, "ind-object"]], "pval2"]),
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

		 # Single pval
		 ("NOM-NP-AS-NP-SC",			["NP", ["IN_as", "NP"]],						["object", "pval"]),
		 ("NOM-NP-AS-NP",				["NP", ["IN_as", "NP"]],						["object", "pval"]),
		 ("NOM-AS-NP",					[["IN_as", "NP"]],								["pval"]),
		 ("NOM-NP-PP",					["NP", "PP"],									["object", "pval"]),

		 # Double objects
		 ("NOM-NP-NP",					["NP", "NP"],									["ind-object", "object"]),

		 # Adverb
		 ("NOM-ADVP-PP",				["ADVP", "PP"],									["adverb", "pval"]),
		 ("NOM-NP-ADVP",				["NP", "ADVP"],									["object", "adverb"]),
		 ("NOM-ADVP",					["ADVP"],										["adverb"]),

		 # Basic
		 ("NOM-PP",						["PP"],											["pval"]),
		 ("NOM-NP",						["NP"],											["object"])
	]

	return comlex_table


def get_pronoun_dict():
	pronoun_dict = {"he":["his", "him"], "she":["her", "her"], "it":["its", "its"], "they":["their", "them"], "we":["our", "us"], "i":["my", "me"]}

	return pronoun_dict




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
		if i >= best_subword_length or (i == best_subword_length and any([possible_word.endswith(end) for end in preferable_endings])):
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


def seperate_line_print(input_to_print):
	if type(input_to_print) == list:
		for x in input_to_print:
			if should_print: print(x)
	elif type(input_to_print) == dict:
		for tag, x in input_to_print.items():
			if should_print: print(str(tag) + ": " + str(x))


def get_all_of_noms(nomlex_entries):
	"""
	Returns a dictionary of all the nominalizations in the given nomlex entries
	:param nomlex_entries: a dictionary of nominalizations
	:return: dictionary of nominalizations (nominalizations: nominalizations_without_numbers)
	"""

	all_noms = {}

	for nom in nomlex_entries.keys():
		all_noms.update({nom: "".join([i for i in nom if not i.isdigit()])})

	return all_noms