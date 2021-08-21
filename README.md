# Whole Community Invasions and the Integration of Novel Ecosystems

This is the repository of code used in the above-titled work (currently under review) by Colin Campbell, Laura Russo, Reka Albert, Angus Buckling, and Katriona Shea.

The code was written for Python 3.x by Colin Campbell. The files (in order of workflow) are as follows:
* **function_library.py**: contains functions referenced in other files. This is a "master" library for this and related work; not all functions enclosed are used for this project.

* **create_graphs.py**: Creates species interaction networks. Generates: *graphlist.data*

* **find_attractors.py**: Identifies the stable communities from the species interaction networks. Uses: *graphlist.data* Generates: *attractors_##.data*

* **invasions_whole.py**: Performs exhaustive invasions between the stable communities. Uses: *graphlist.data*, *attractors_##.data* Generates: *attractors_comb_whole_##.data*

* **invasions_random.py**: Performs exhaustive invasions between stable communities and randomly selected invaders. Uses: *graphlist.data*, *attractors_##.data* Generates: *attractors_comb_random_##.data*

* **analysis.py**: Analyzes the output of the two invasions files. Generates Figures 3-5 in the manuscript and reports output of statistical tests. Uses: *graphlist.data*,  *attractors_comb_whole_##.data*,  *attractors_comb_random_##.data*

Questions can be directed to Colin Campbell at campbeco@mountunion.edu.
