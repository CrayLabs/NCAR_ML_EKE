

[![Input Dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4682270.svg)](https://doi.org/10.5281/zenodo.4682270)

# Using Machine Learning at Scale in HPC Simulations with SmartSim

This project is code for the paper: Using Machine Learning at Scale in HPC Simulations with SmartSim

# Reproduction

We strove to make this project as reproducible as possible. If you find any places where
reproducibility is lacking or could be improved, please file an issue to let us know.

## Installation

Below are the steps to install the various components needed to reproduce the
results in the paper for the ML-EKE parameterization with SmartSim

### Clone

First, clone the repository with the MOM6 submodule

```bash
git clone --recursive https://github.com/CrayLabs/NCAR_ML_EKE.git
```

### Install SmartSim

See [documentation](https://www.craylabs.org/docs/installation.html) for full build instructions if necessary

Update: Please use SmartRedis 0.2.0 and SmartSim 0.3.2 if available (Release: July 30, 2021)
Otherwise, use branch develop for both libraries.

```bash
pip install smartsim==0.3.1 # 0.3.2 if available
export CUDNN_LIBRARY=/path/to/cudnn/library
export CUDNN_INCLUDE_DIR=/path/to/cudnn.h
smart --device gpu --no_tf # just build the PyTorch backend. Use -v for verbose mode
```

### Install and Build SmartRedis

Build SmartRedis from source to compile into MOM6

```text

Modules used in paper

Currently Loaded Modulefiles:
  1) modules/3.2.11.4                                    14) Base-opts/2.4.142-7.0.3.0_42.11__g8f27585.ari
  2) craype-network-aries                                15) cray-mpich/7.7.18.1
  3) nodestat/2.3.89-7.0.3.0_33.12__g8645157.ari         16) dws/3.0.36-7.0.3.0_65.9__g6985c90.ari
  4) sdb/3.3.818-7.0.3.0_26.26__g8ad6d1f.ari             17) craype/2.7.10.1
  5) udreg/2.3.2-7.0.3.0_36.21__g5f0d670.ari             18) cray-libsci/20.09.1
  6) ugni/6.0.14.0-7.0.3.0_25.20__gdac08a5.ari           19) pmi/5.0.17
  7) gni-headers/5.0.12.0-7.0.3.0_37.17__gd0d73fe.ari    20) atp/3.13.1
  8) dmapp/7.1.1-7.0.3.0_38.29__g93a7e9f.ari             21) rca/2.2.20-7.0.3.0_24.22__g8e3fb5b.ari
  9) xpmem/2.2.27-7.0.3.0_47.2__gada73ac.ari             22) perftools-base/21.05.0
 10) llm/21.4.632-7.0.3.0_44.6__gf148da5.ari             23) PrgEnv-gnu/6.0.10
 11) nodehealth/5.6.28-7.0.3.0_75.26__g742816f.ari       24) cray-netcdf/4.7.4.4
 12) system-config/3.6.3181-7.0.3.0_50.1__g4e5190fd.ari  25) cray-hdf5/1.12.0.4
 13) slurm/20.11.5-1                                     26) gcc/8.3.0
```

See [documentation](https://www.craylabs.org/docs/installation.html) for full build instructions if necessary

Note: There is a ``env`` file in ``MOM6/build/gnu`` that specifies
the programming environment we built with. Specifically, we used
the GNU toolchain with gcc 8.3.1.

 > IMPORTANT: Source the env script before building anything (if you are on a Cray or HPC system with modules)

```bash
git clone https://www.github.com/CrayLabs/SmartRedis.git smartredis
cd smartredis
# checkout the 0.2.0 tag if available otherwise use develop
make lib
export SMARTREDIS_INSTALL_PATH=$(pwd)/install
```

### Build MOM6

Assuming you checked out the repository, built and installed
SmartSim and SmartRedis, you now need to build MOM6.

Follow the [Getting Started](https://github.com/NOAA-GFDL/MOM6-examples/wiki/Getting-started)
portion of the MOM6-examples wiki for compiling and running the
MOM6-SIS2 coupled model. Please be sure to replace the MOM6 directory
from this repository in MOM6-examples/src/MOM6

In the
[Downloading input data](https://github.com/NOAA-GFDL/MOM6-examples/wiki/Getting-started#downloading-input-data)
section, make sure to download the ``OM4_025``, ``obs``, and ``CORE``
directories.

### Download the SmartSim-MOM6 input data

We host and include the input data we used to run MOM6 along with
pre-trained models and scripts we used for the paper.

To download the data, either at the DOI link at the top
of the repo or [here](https://doi.org/10.5281/zenodo.4682270)

Download the data into ``MOM6/INPUT``. The MOM6 input dataset
pretrained models and scripts for the SmartSim workload are all
included. Replace the hidden ``.datasets`` symlink to the directory
where you downloaded the MOM6 input data.

Copy the executable built previously into the MOM6 directory.

## Run

Before running the SmartSim driver script, be sure that
the computational setup described by the script suits your
system. 

This script assumes launching on a slurm cluster
with at least
   - 228 CPU nodes with 96 cpus (including hyperthreads)
   - 16 nodes with P100 GPUs and 36 cpu cores (including hyperthreads)

This can be changed to suit your system with the parameters
listed below

To run the exact same experiment as our paper, increase
the time in both batch jobs and the number of days
to 10 years. This is hopefully obvious how to do in the
script.

Once configured, the entire workload can be executed with

```bash
# make sure python environment with SmartSim installed is active
cd driver
python driver.py
```

Note: this will submit two batch allocations to the scheduler
of large size if configurations are not changed. To add account
or other information please consult the [SmartSim API Docs](https://www.craylabs.org/build/html/api/smartsim_api.html#smartsim-api)

## Results

When the workload is run successfully, there will be a 
``AI-EKE-MOM6`` directory with all of the output from each
ensemble member. Included are timings in each ``MOM_<ensemble_number>.out``
which look like

```text
(SmartRedis put tensor)               0.003483      0.019676      0.010159      0.003618  0.000    41     0   909
(SmartRedis run model)                0.946908      2.807891      1.943015      0.462866  0.008    41     0   909
(SmartRedis unpack tensor )           0.001292      0.011080      0.001756      0.000605  0.000    41     0   909
```
You can use these to examine the overall timings of each operation that uses 
SmartRedis inside MOM6

## Variants

Below are some variants that can be run for examining different
configurations or for different systems

### Reference Simulation

The compare the SmartSim approach vs the MEKE paramterization, 
change the line in ``MOM6/OM4_025/MOM_override`` from

``EKE_SOURCE='sr'``
to 
``EKE_SOURCE='prog'``

Then comment out the parts of the driver script that create
and launch the database. Once commented out, you can run the
driver script as normal and the MOM6 simulations will be
executed with the MEKE paramterization instead of the
Smartsim approach.

### CPU-only Machines

If you don't have GPU nodes, don't worry, we have you covered.
Follow the instructions to install SmartSim for CPU. We include
a pre-trained model for CPU inference as well. To use the CPU
model, change the line in ``MOM6/OM4_025/MOM_override`` from

``SMARTREDIS_MODEL='ncar_ml_eke.gpu.pt'``
to
``SMARTREDIS_MODEL='ncar_ml_eke.pt'``

both models are included in the input data directory you downloaded
earlier.


## Contributors

The collaboration was a joint effort between Hewlett Packard Enterprise (HPE),
National Center for Atmosheric Research (NCAR), and the University of Victoria (U Vic)

Contributors in no particular order

 - Andrew Shao (U Vic)
 - Sam Partee (HPE)
 - Alessandro Rigazzi (HPE)
 - Scott Bachman (NCAR)
 - Gustavo Marques (NCAR)
 - Matthew Ellis (HPE)
