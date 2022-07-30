import pickle

from yet_another_verb.data_handling.bytes.bytes_handler import BytesHandler
from yet_another_verb._bw_alias import *


class PKLBytesHandler(BytesHandler):
	@staticmethod
	def loads(bytes_data: bytes) -> object:
		return pickle.loads(bytes_data)

	@staticmethod
	def saves(data: object) -> bytes:
		return pickle.dumps(data)
