## Model creation examples

* `prepare_test_data.sh` - prepare synthetic dataset
* `prepare_test_data_with_bias.sh` - prepare synthetic dataset with some bias field

To execute examples, setup PYTHONPATH to point to the location of `ipl` directory, then run `python -m scoop -n <n> <script.py>`

* `scoop_test_1.py` - create 3 symmetric models, using linear registration with 6 parameters (rigid body), 9 parameters and 12 parameters
* `scoop_test_2.py` - create 3 non-symmetric models, using linear registration with 6 parameters (rigid body), 9 parameters and 12 parameters
* `scoop_test_bias*.py` - generate models compensating for bias field ( linear coregistration)
* `scoop_test_nl.py` - generate model using non-linear coregistration with minctracc, non-simmetric
* `scoop_test_nl_sym.py` - generate model using non-linear coregistration with minctracc, simmetric
* `scoop_test_nl_elastix.py` - generate model using non-linear coregistration with Elastix
* `scoop_test_nl_ants.py` - generate model using non-linear coregistration with ANTs
* `scoop_test_nl_dd.py` - generate model using non-linear coregistration with diffeomorphic demons
* `scoop_test_nl_ldd.py` - generate model using non-linear coregistration with log-diffeomorphic demons
