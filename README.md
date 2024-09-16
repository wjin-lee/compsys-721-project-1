# COMPSYS 721 - Project 1 - wlee447

> NOTE: This was run in Ubuntu 22.24 using Python 3.10 via WSL2 (with NVIDIA GPU passthrough). **I have not tested with any other combination!**. *Here be dragons!*  

## Layout Description
- `requierments.txt` outlines the dependencies required to run the script.
- `result.json` During hyperparameter tuning, the results will get saved here as we go along in case the script crashes or we run out of memory.
- `load_lfw.py` defines utility functions to load the Labelled Faces in the Wild (LFW) dataset. The path to this dataset (from which all class folders are visible) can be configured in `main.py`.
- `main.py` The main script responsible for training as well as evaluation. Near the top of the file, it contains various constant options that can be modified. (E.g. the root path to LFW dataset, and the seed).

## main.py
After setting up dependencies, the script can be run like so:
```bash
# Setup Python 3.10, pip install -r requirements.txt, other setup calls (if using conda), etc... 
python ./main.py
```

The `main()` function can be modified for various tasks. For hyperparmeter tuning ResNet for example, the gridsearch section for just ResNet can be uncommented.

Dividers mark section boundaries:
```python
#============= DIVIDER ===========
... Code here ...
#=================================
```

See function comments for detailed documentation.

Thank you!