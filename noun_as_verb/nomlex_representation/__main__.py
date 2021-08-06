from lisp_to_json.lisp_to_json import lisps_to_jsons
from nomlex_adaptation.lexicon_adaptation import generate_adapted_lexicon

lisp_file_path = "/Users/avivwn/Documents/University/Bar Illan University (M.Sc)/Courses/Thesis/nomlex/noun_as_verb/nomlex_representation/lexicons/lisp_lexicons/NOMLEX-plus.1.0.txt"

with open(lisp_file_path, "r") as f:
	lisp_text = " ".join(f.read().splitlines())

json_data = lisps_to_jsons(lisp_text)

a = generate_adapted_lexicon(json_data)
print("something")

import numpy as np
subcat_types = []
subcat_count = []
in_subcat_structures_count = []
in_structure_constraints_count = []
entry_maps_count = []
for nom in a.entries.values():
	subcat_count.append(len(nom.subcats))
	subcat_types += list(nom.subcats.keys())
	c = 0
	for subcat in nom.subcats.values():
		in_subcat_structures_count.append(len(subcat))
		for structure in subcat:
			in_structure_constraints_count.append(len(structure.sub_constraints))
			c += len(structure.sub_constraints)
	entry_maps_count.append(c)
print(np.max(entry_maps_count), np.min(entry_maps_count), np.mean(entry_maps_count))
print(np.mean(subcat_count))
print(np.mean(in_subcat_structures_count))
print(np.mean(in_structure_constraints_count))
