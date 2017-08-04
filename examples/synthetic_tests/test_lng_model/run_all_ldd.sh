#! /bin/sh

python -m scoop -n 4 test_lng_model_LCC_ldd.py
python -m scoop -n 4 test_model_std.py
python -m scoop -n 4 test_model_SSD_ldd.py
python -m scoop -n 4 test_lng_model_LCC_ldd.py
