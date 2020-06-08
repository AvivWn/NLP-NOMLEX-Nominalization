from .lisp_to_json import *

if __name__ == '__main__':
	"""
	Command line arguments- 
		lisp_file_name (only the file name, without its all path)
	"""
	import sys

	if not os.path.exists(JSON_DIR):
		os.makedirs(JSON_DIR)

	lisp_to_json(sys.argv[1])

	#print(list(set(phrases)))