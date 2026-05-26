# Automated reading of vertebral images for aging shortfin mako (Isurus oxyrinchus) using Bayesian deep learning

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20401378.svg)](https://doi.org/10.5281/zenodo.20401378)

Code and data for the publication: Automated reading of vertebral images for aging shortfin mako (Isurus oxyrinchus) using Bayesian deep learning

## Directory structure

### Python scripts

* [preprocessing_scripts](./preprocessing_scripts)
  * [preprocessing_scripts/1_convert.py](./preprocessing_scripts/1_convert.py): Convert JPEG files to PNG and remove file name prefix.
  * [preprocessing_scripts/2_replace_char_imgs.py](./preprocessing_scripts/2_replace_char_imgs.py): Filter out non-vertebrae images.
  * [preprocessing_scripts/3_process_individuals.py](./preprocessing_scripts/3_process_individuals.py): Process each vertebrae image by resizing, sharpening image quality, and centering images.
  * [preprocessing_scripts/4_black_and_white.py](./preprocessing_scripts/4_black_and_white.py): Convert images to black and white (not used).
  * [preprocessing_scripts/5_get_image_ages.py](./preprocessing_scripts/5_get_image_ages.py): Create the [image_ages.csv](./data/image_ages.csv) file.
  * [preprocessing_scripts/6_augment_imgs.py](./preprocessing_scripts/6_augment_imgs.py): Augment the size of the training dataset.
* [training_scripts](./training_scripts)
  * [training_scripts/1_image_regression_nll_torch.py](./training_scripts/1_image_regression_nll_torch.py): Train one ResNet model using Negative Log Likelihood.
  * [training_scripts/2_image_regression_ensembling_nll_torch.py](./training_scripts/2_image_regression_ensembling_nll_torch.py): Train a DeepEnsemble using Negative Log Likelihood.
  * [training_scripts/3_image_regression_ensembling_nll_uncertainty.py](./training_scripts/3_image_regression_ensembling_nll_uncertainty.py): Quantify the uncertainty of a DeepEnsemble using Negative Log Likelihood.
  * [training_scripts/4_image_regression_mola_nll_torch.py](./training_scripts/4_image_regression_mola_nll_torch.py): Train a Mixture of Laplace Approximations (MoLA) using Negative Log Likelihood.
  * [training_scripts/5_image_regression_mola_nll_uncertainty.py](./training_scripts/5_image_regression_mola_nll_uncertainty.py): Quantify the uncertainty of a Mixture of Laplace Approximations (MoLA) using Negative Log Likelihood.
  * [training_scripts/6_image_regression_swag_nll_torch.py](./training_scripts/6_image_regression_swag_nll_torch.py): Train Multiple Stochastic Weight Averaging-Gaussian (Multi-SWAG) using Negative Log Likelihood.
  * [training_scripts/7_image_regression_nll_swag_uncertainty.py](./training_scripts/7_image_regression_nll_swag_uncertainty.py): Quantify the uncertainty of Multiple Stochastic Weight Averaging-Gaussian (Multi-SWAG) using Negative Log Likelihood.
  * [training_scripts/8_image_regression_interpretability_torch.py](./training_scripts/8_image_regression_interpretability_torch.py): Utilise interpretability algorithms using the `captum` Python library.

### Data

* [data](./data): Tabular data files for images and their band counts.
  * [data/sb_ages](./data/sb_ages): The raw vertebrae band counts.
  * [data/image_ages](./data/image_ages): Vertebrae image path information and their respective band counts.
* [in](./in): The image input file directory.
* [out](./out): The image output file directory.

### External

* [swag_experiments](./swag_experiments): The code in this subdirectory was ported from https://github.com/izmailovpavel/understandingbdl/tree/master
