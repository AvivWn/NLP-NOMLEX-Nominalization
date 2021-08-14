import inflect
from langdetect import detect, DetectorFactory, lang_detect_exception

DetectorFactory.seed = 42
engine = inflect.engine()


def in_plural(word: str):
	return engine.plural(word)


def is_english(text: str) -> bool:
	try:
		lang = detect(text)
		return lang == "en"
	except lang_detect_exception:
		return False


def clean_sentence(sent: str) -> str:
	# Avoiding underscore, cause it appears a lot on the wikipedia dataset
	sent = sent.replace("_", " ").replace("\t", "").strip(" \t\r\n")

	# Replace multi-whitespaces to a single one
	sent = ' '.join(sent.split())

	return sent
