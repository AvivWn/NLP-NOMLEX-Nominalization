import sys
import spacy
import pickle
spacy.prefer_gpu()
nlp = spacy.load('en_core_web_sm')



def clean_sentence(sent):
	"""
		Cleans a given sentence from starting with space or ending with new line sign
		:param sent: a sentece (string)
		:return: the cleaned sentence (as string)
	"""

	while sent.startswith(" "):
		sent = sent[1:]

	sent = sent.replace("\n", "").replace("\r\n", "").replace("\r", "")

	return sent

def get_dependency(sent):
	"""
	Returns the dependency tree of a given sentence
	:param sent: a string sentence
	:return: the dependency tree of the sentence (a list of tuples)
	"""

	dep = []

	sentence_info = nlp(sent)
	for word_info in sentence_info:
		head_id = str(word_info.head.i + 1)  # we want ids to be 1 based
		if word_info == word_info.head:  # and the ROOT to be 0.
			assert (word_info.dep_ == "ROOT"), word_info.dep_
			head_id = "0"  # root

		str_sub_tree = " ".join([node.text for node in word_info.subtree])
		dep.append(
			"\t".join((str(word_info.i + 1), str(word_info.text), str(word_info.lemma_),
			 str(word_info.tag_), str(word_info.pos_), str(head_id),
			 str(word_info.dep_), str(word_info.ent_iob_), str(word_info.ent_type_), str_sub_tree)))

	return dep

def parse_sentences(sentences_file_name):
	sentences_file = open(sentences_file_name, "r")
	#output_file = open('x.pkl', 'wb')
	output_file = open(sentences_file_name + ".parsed", "w")
	log_file = open("log", "a+")

	for i, sent in enumerate(sentences_file.readlines()):
		sent = clean_sentence(sent)
		#pickle.dump(get_dependency(sent), output_file)
		#pickle.dump("", output_file)

		print("\n".join(get_dependency(sent)), file=output_file)
		print("", file=output_file)
		print(sentences_file_name + ".parsed: ", i, file=log_file, flush=True)

	output_file.close()
	sentences_file.close()

if __name__ == '__main__':
	sentences_file_name = sys.argv[1]
	parse_sentences(sentences_file_name)