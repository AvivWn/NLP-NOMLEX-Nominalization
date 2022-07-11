import os.path

from pony.orm import Database

from yet_another_verb.data_handling.db.communicators.communicator import DBCommunicator


class SQLiteCommunicator(DBCommunicator):
	def __init__(self, db: Database, db_path: str):
		self.db = db
		self.db_path = db_path

		self.connect()

	def connect(self):
		should_create_db = not os.path.exists(self.db_path)
		self.db.bind(provider='sqlite', filename=self.db_path, create_db=should_create_db)

	def disconnect(self):
		self.db.disconnect()

	def generate_mapping(self):
		if self.db.schema is None:
			self.db.generate_mapping(create_tables=True)

	def commit(self):
		self.db.commit()
