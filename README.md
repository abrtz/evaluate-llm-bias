# Evaluate LLM Bias by Replicating MBBQ

This repository evaluates Llama 2 7B by replicating the study of Neplenbroek et al. (2024) and extends the scope to nationality, gender and their intersection.

## Requirements

Local experiments were run in Python 3.11.8.
All required libraries and versions are provided in the requirements.txt file. Run the following command to install them.

`pip install -r requirements.txt`

## README

1. The datasets were created by mimicking the approach and reclycling the code of Parrish et al. (2022) 

First, the most common names were gathered from Argentine Civil Registry with the notebook `names_arg.ipynb`

Then, the datasets were created by running the scripts: 
* generate_from_template_all.py 
* generate_from_template_intersection.py 

as in `python script_name.py` (add `--control True` to create control datasets.)

These scripts access the templates from the templates directory (hereby saved as password-protected .zip files. The password is mbbq). They create the datasets that are saved under the `/data` directory. Place the gender datasets (control and biased) from Neplenbroek et al. (2024) (available at https://github.com/Veranep/MBBQ/tree/main/data) in this directory as well.

2. MBBQ (Neplenbroek et al., 2024) was replicated for Llama 2 7B in `replicate_and_extend_mbbq.ipynb`. This notebook was run Google Colab with an NVIDIA A100 GPU.

The resulting files were stored in `llama_responses.zip`. Password is the same as mentined aboved.

3. The bias and accuracy metrics are computed by running `python compute_metrics.py`

4. An error analysis was carried in the notebook `error_analysis.ipynb`.




