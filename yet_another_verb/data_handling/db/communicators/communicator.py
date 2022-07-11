import abc


class DBCommunicator(abc.ABC):
	@abc.abstractmethod
	def connect(self):
		pass

	@abc.abstractmethod
	def disconnect(self):
		pass

	@abc.abstractmethod
	def generate_mapping(self):
		pass

	@abc.abstractmethod
	def commit(self):
		pass
