from enum import Enum


class DepRelation(str, Enum):
	NMOD = "nmod"
	NSUBJ = "nsubj"
	NSUBJPASS = "nsubjpass"
	DOBJ = "dobj"
	IOBJ = "iobj"
	ADVMOD = "advmod"
	AMOD = "amod"
	ADVCL = "advcl"
	XCOMP = "xcomp"
	CCOMP = "ccomp"
	ACL = "acl"
	ACL_RELCL = "acl:relcl"
	NMOD_POSS = "nmod:poss"
	COMPOUND = "compound"
	COP = "cop"
	PRT = "prt"
	COMPOUND_PRT = "compound:prt"
	MARK = "mark"
	CASE = "case"
	MWE = "mwe"
	AUXPASS = "auxpass"


DEP_RELATIONS = {r for r in DepRelation}
PREP_RELATIONS = [DepRelation.CASE, DepRelation.MARK]

# backward compatability object name
WordRelation = DepRelation
