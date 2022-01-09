class DemoConfig:
	def __init__(
			self,
			data_dir="data",
	):
		self.URL = "127.0.0.1"  # '0.0.0.0'
		self.PORT = 5000

		self.DATA_DIR = data_dir
		self.EXAMPLE_DATA_PATH = self.DATA_DIR + "/examples.txt"
		self.PARSED_EXAMPLE_DATA_PATH = self.DATA_DIR + "/examples.parsed"

		self.MIN_RESPONSE_TIME = 1
		self.MAX_MATCHING_RESULTS = 5


DEMO_CONFIG = DemoConfig()
