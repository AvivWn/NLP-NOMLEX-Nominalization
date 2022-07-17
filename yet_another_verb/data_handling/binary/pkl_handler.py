import pickle

from yet_another_verb._bw_alias import *


class PKLHandler:
	@staticmethod
	def loads(data_bytes: bytes) -> object:
		return pickle.loads(data_bytes)

	@staticmethod
	def dumps(data: object) -> bytes:
		return pickle.dumps(data)
