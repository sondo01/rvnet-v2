# rv_net

This is a forge of Zoe de Beurs' rv_net repository https://github.com/zdebeurs/rv_net.git.
For more information, please refer to the original repository.

## Installation

1. Clone repo:

    ```
    git clone https://github.com/sondo/rvnet-v2.git
    ```

    At this point, the rvnet-v2 directory should be created

2. Create a virtual environment with Python 3.11 (instead of using complex conda) and Install dependencies:

    ```
    $ cd rvnet-v2
    $ python3.11 -m venv .venv
    $ source .venv/bin/activate
    $ pip install -r requirements.txt
    ```

3. We need to install mpyfit: <https://github.com/evertrol/mpyfit>

    - For MAC: run 
        ``` 
        $ pip install -e . 
        ``` 
        (don't forget . at the end)

    - For Linux:
        ```
        $ git clone https://github.com/evertrol/mpyfit
        $ cd mpyfit
        $ python setup.py install --user
        $ python setup.py build_ext --inplace
        $ cp -r mpyfit ../
        ```

    After this step, we need to check if the **mpyfit** directory exists in the rvnet-v2 directory. The mpyfit directory must have a __init__.py file

## Download Data from DACE

4. Download fits files from DACE using *dace_query_fits.py*

    ```
    $ python dace_query_fits.py
    ```

5. Download *public_release_timeseries.csv* files from DACE using *dace_query_public_release.py*

    ```
    $ python dace_query_public_release.py
    ```

6. Because filename in *public_release_timeseries.csv* is not in the same format as fits files, we need to correct it

    ```
    $ python correct_release_filename.py
    ```

## Preprocess raw data

7. Clean up bad observations (cloud-free <99% or rv_diff_extinction < 0.1m/s)

    ```
    $ python clean_up.py
    ```

    All bad observations will be changed to *.stif* and so will be ignored in the next steps

- Note: to reverse the change, run:

    ```
    $ ./stif2fits.sh
    ```

8. Create NPZ files from raw data

    ```
    python PrepareData.py
    ```

    This script creates numpy files (in *.npz* format) ready for the next steps:

9.  Create TF files:

    Run **Making_TF_records_tutorial.ipynb**

10. Train model: 

    Run **training.ipynb**

## Note: 
- *training.ipynb* is a Tensorflow v2 version of *2_3_1_HARPS_Linear_FC_CN_June10_2023.ipynb*. (Tested on Debian Linux 12, with Python 3.11)
- *master_shifting.py* was modified to work with the new 49 pixel CCFs

