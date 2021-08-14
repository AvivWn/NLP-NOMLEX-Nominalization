from enum import Enum


class POSTag(str, Enum):
	CC = "CC"			# Coordinating conjunction
	CD = "CD"			# Cardinal number
	DT = "DT"			# Determiner
	EX = "EX"			# Existential there
	FW = "FW"			# Foreign word
	IN = "IN"			# Preposition or subordinating conjunction
	JJ = "JJ"			# Adjective
	JJR = "JJR"			# Adjective, comparative
	JJS = "JJS"			# Adjective, superlative
	LS = "LS"			# List item marker
	MD = "MD"			# Modal
	NN = "NN"			# Noun, singular or mass
	NNS = "NNS"			# Noun, plural
	NNP = "NNP"			# Proper noun, singular
	NNPS = "NNPS"		# Proper noun, plural
	PDT = "PDT"			# Predeterminer
	POS = "POS"			# Possessive ending
	PRP = "PRP"			# Personal pronoun
	PRP_POSS = "PRP$"   # Possessive pronoun
	RB = "RB"			# Adverb
	RBR = "RBR"			# Adverb, comparative
	RBS = "RBS"			# Adverb, superlative
	RP = "RP"			# Particle
	SYM = "SYM"			# Symbol
	TO = "TO"			# to
	UH = "UH"			# Interjection
	VB = "VB"			# Verb, base form
	VBD = "VBD"			# Verb, past tense
	VBG = "VBG"			# Verb, gerund or present participle
	VBN = "VBN"			# Verb, past participle
	VBP = "VBP"			# Verb, non-3rd person singular present
	VBZ = "VBZ"			# Verb, 3rd person singular present
	WDT = "WDT"			# Wh-determiner
	WP = "WP"			# Wh-pronoun
	WP_POSS = "WP$"		# Possessive wh-pronoun
	WRB = "WRB"			# Wh-adverb


POSTAGS = {t for t in POSTag}
NOUN_POSTAGS = [POSTag.NN, POSTag.NNS, POSTag.NNP, POSTag.NNPS, POSTag.PRP]
VERB_POSTAGS = [POSTag.VB, POSTag.VBD, POSTag.VBG, POSTag.VBN, POSTag.VBP, POSTag.VBZ]
ADVERB_POSTAGS = [POSTag.RB, POSTag.RBR, POSTag.RBS]
ADJECTIVE_POSTAGS = [POSTag.JJ, POSTag.JJR, POSTag.JJS]
