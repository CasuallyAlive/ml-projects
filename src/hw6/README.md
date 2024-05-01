# HW2 Submission
## Description
Homework 6 submission for ML. The 'report/' directory contains the source latex files of the written section and the 'data/' contains the directory for the data files. I drafted a jupyter notebook to run and organize my experiments. The file can be found [here](./hw6_exp_report.ipynb). All imported modules should be available on the cade "base" conda environment. Please make sure to select the base environment as your jupyter kernel to run my implementation and the necessary experiments. My SVM implementation can be found [here](./svm.py) and my logistic regression classifier [here](./logistic_reg.py). Please make sure to keep my [utils](./utils.py) file in the same directory as my notebook since it contains multiple functions that I used and implemented for the project.

## Troubleshooting
* Import Error - try adding jupyter notebook src directory to python path.
    ```python
    import sys
    sys.path.append('<path-to-folder>/src')
    ```
