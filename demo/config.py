class DemoConfig:
	def __init__(
			self,
			data_dir="data",
	):
		self.URL = '0.0.0.0'  # "127.0.0.1"
		self.PORT = 5000
		self.WWW_ENDPOINT = "NounVerbUDTransfer"

		self.DATA_DIR = data_dir
		self.EXAMPLE_DATA_PATH = self.DATA_DIR + "/examples_from_paraphrasing.txt"
		self.PARSED_EXAMPLE_DATA_PATH = self.DATA_DIR + "/examples_from_paraphrasing.parsed"

		self.MIN_RESPONSE_TIME = 1
		self.MAX_MATCHING_RESULTS = 5


DEMO_CONFIG = DemoConfig()
