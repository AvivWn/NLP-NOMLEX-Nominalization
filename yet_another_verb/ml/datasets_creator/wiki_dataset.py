from tqdm import tqdm
from nlp import load_dataset
from nltk.tokenize import sent_tokenize
from itertools import chain

from .dataset_creator import DatasetCreator
from yet_another_verb.file_handlers.txt_file_handler import TXTFileHandler
from yet_another_verb.utils.linguistic_utils import clean_sentence


class WikiDatasetCreator(DatasetCreator):
	def __init__(self, dataset_size=None, **kwargs):
		super().__init__(dataset_size)

	def _articles_to_sents(self, wiki_articles):
		total_sents = []

		for article_info in tqdm(wiki_articles):
			text = article_info["text"]
			paras = text.split("\n_START_PARAGRAPH_\n")

			for para in paras:
				# Avoid article titles
				if "_START_ARTICLE_" in para:
					continue

				# Avoid section titles
				sents_str = para.split("_START_SECTION_")[0]
				sents_str = sents_str.replace("_NEWLINE_", "\n")
				sents = sents_str.split("\n")
				sents = chain(*[sent_tokenize(sent) for sent in sents])
				sents = [clean_sentence(sent) + "\n" for sent in sents if sent != ""]
				total_sents += sents

			if self.has_reached_size(total_sents):
				break

		return total_sents

	def create_dataset(self, out_dataset_path):
		# Extract all the sentences from the wikipedia articles
		wiki_datasets = load_dataset("wiki40b", "en")
		train_sents = self._articles_to_sents(wiki_datasets["train"])
		val_sents = self._articles_to_sents(wiki_datasets["validation"])
		test_sents = self._articles_to_sents(wiki_datasets["test"])
		out_dataset = train_sents + val_sents + test_sents

		TXTFileHandler().save(out_dataset_path, out_dataset)
