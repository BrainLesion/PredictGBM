# Predict-GBM

[![Python Versions](https://img.shields.io/pypi/pyversions/predict-gbm)](https://pypi.org/project/predict-gbm/)
[![Stable Version](https://img.shields.io/pypi/v/predict-gbm?label=stable)](https://pypi.python.org/pypi/predict-gbm/)
[![Documentation Status](https://readthedocs.org/projects/predict-gbm/badge/?version=latest)](http://predict-gbm.readthedocs.io/?badge=latest)
[![tests](https://github.com/BrainLesion/PredictGBM/actions/workflows/tests.yml/badge.svg)](https://github.com/BrainLesion/PredictGBM/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/BrainLesion/predict-gbm/graph/badge.svg?token=A7FWUKO9Y4)](https://codecov.io/gh/BrainLesion/predict-gbm)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Providing a unified framework for evaluating and benchmarking glioblastoma models by assessing radiation plan coverage on recurrences observed in follow-up MRI exams.
## Features
- Easy-to-use image preprocessing and model evaluation pipeline that handles everything from DICOM conversion to registration, model prediction, radiation plan generation and evaluation.
- Dockered versions of recent glioblastoma growth models and instructions on including novel methods
- Access to a preprocessed glioblastoma dataset comprised of a few hundred subjects


## Installation

Prerequisites:
- **Docker**: Installation instructions on the official [website](https://docs.docker.com/get-docker/)
- **NVIDIA Container Toolkit**: Refer to the [NVIDIA install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and the official [GitHub page](https://github.com/NVIDIA/nvidia-container-toolkit)
- **dicom2niix**: Required if you plan to process raw DICOM data.

A pypi package will be available soon. Currently, the package can be installed with poetry:

```bash
curl -sSL https://install.python-poetry.org | python3 -
git clone https://github.com/BrainLesion/PredictGBM
cd PredictGBM
poetry install
```

## Data and Models

Preprocessed data can be obtained from [TODO]()

Ready-to-use dockered versions are available for some growth models from [TODO](). Placing them in `predict_gbm/models/` just like the `test_model.tar`, allows you to use them via `algo_id="test_model"`. 

## Use Cases and Tutorials

Examples can be found in `/scripts`:

- [single_dicom.py](scripts/single_dicom.py) shows how to quickly process a single patient with DICOM files.
- [single_nifti.py](scripts/single_nifti.py) shows how to quickly process a single patient with NIfTI files.
- [dataset_example.py](scripts/dataset_example.py) shows how to use the PatientDataset to parse datasets.
- [stepwise_processing.py](scripts/stepwise_processing.py) shows how to run standalone pipeline components.
- [evaluate_predict_gbm.py](scripts/evaluate_predict_gbm.py) shows how to evaluate on the PredictGBM dataset.


## Adding new growth models

This repository can be used to perform inference or benchmark with your own tumor growth model. To this end, you need to create a docker image of your growth model. The following sections serve as guideline on how the image should be created. 

### Directory structure

Input and output data are passed to/from the container using mounted directories:

**Input:**

```bash
/mlcube_io0
   ┗ Patient-00000
      ┣ 00000-gm.nii.gz
      ┣ 00000-wm.nii.gz
      ┣ 00000-csf.nii.gz
      ┣ 00000-tumorseg.nii.gz
      ┗ 00000-pet.nii.gz
```

**Output:**

```bash
/mlcube_io1
   ┗ 00000.nii.gz
```

### Dockerfile Example

Ensure the container adheres to the above I/O structure. An example Dockerfile could be:

```dockerfile
# Image and environment variables
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir --upgrade pip

WORKDIR /app

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code to workdir
COPY . .
ENTRYPOINT ["python3", "inference.py"]
```

## Citation

If you use PredictGBM in your research, please cite it to support the development!

```
TODO: citation will be added asap
```

### Reporting Bugs, Feature Requests and Questions

Please open a new issue [here](https://github.com/BrainLesion/PredictGBM/issues).

### Code contributions

Nice to have you on board! Please have a look at our [CONTRIBUTING.md](CONTRIBUTING.md) file.
