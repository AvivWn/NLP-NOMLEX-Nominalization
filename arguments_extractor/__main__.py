from .arguments_extractor import *
from .utils import *

test_extractor = ArgumentsExtractor("NOMLEX-plus.1.0.txt")
test_extractor.extract_arguments = timeit(test_extractor.extract_arguments)
separate_line_print(test_extractor.extract_arguments("His solicitation of Mayor Koch to lead the parade."))
separate_line_print(test_extractor.extract_arguments("Apple appoint Alice."))
separate_line_print(test_extractor.extract_arguments("The appointment of Tim Cook by Apple as a CEO was expected."))
#separate_line_print(test_extractor.extract_arguments("Paris's destruction"))