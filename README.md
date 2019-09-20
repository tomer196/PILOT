# PILOT: Physics-Informed Learned Optimal Trajectories for Accelerated MRI

This repository contains a PyTorch implementation of the paper:

[PILOT: Physics-Informed Learned Optimal Trajectories for Accelerated MRI](https://arxiv.org/abs/1909.05773).

[Tomer Weiss](tomer196@gmail.com), Ortal Senouf, Sanketh Vedula, Oleg Michailovich, Michael Zibulevsky, Alex Bronstein

![teaser](teaser.png)

## Introduction

Magnetic Resonance Imaging (MRI) has long been considered to be among “the gold standards” of diagnostic medical imaging. The long acquisition times, however, contribution to the relative high costs of MRI examination. Over the last few decades, multiple studies have focused on the development of both physical and post-processing methods for accelerated acquisition of MRI scans. In this work, we
propose a novel approach to the learning of optimal schemes for conjoint acquisition and reconstruction of MRI scans, with
the optimization carried out simultaneously. To be of a practical value, the schemes are encoded
in the form of general k-space trajectories, whose associated magnetic gradients are constrained to obey a set of predefined
hardware requirements (as defined in terms of, e.g., peak currents and maximum slew rates of magnetic gradients). We demonstrate the effectiveness of the proposed solution in application to both image reconstruction and image segmentation, reporting substantial improvements in terms of acceleration factors as well as the quality of these end tasks.

This repo contains the codes to replicate our experiment for reconstruction.

## Dependencies

To install other requirements through `$ pip install -r requirements.txt`.

## Usage

First you should download the singelcoil dataset from [fastMRI](https://fastmri.med.nyu.edu/) and split the training + validation sets to training + validation + test set.
Update the datasets location in `common/arg.py`.
We provide scripts to run few experiment, fill free to change the parameters as needed. 

```bash
sh exp_PILOT_EPI.sh         # PILOT with EPI initialization
sh exp_PILOT_spiral.sh      # PILOT with spiral initialization
sh exp_PILOT-TSP.sh         # PILOT-TSP
```

## Citing this Work
Please cite our work if you find this approach useful in your research:
```latex
@ARTICLE{weiss2019pilot,
       author = {{Weiss}, Tomer and {Senouf}, Ortal and {Vedula}, Sanketh and
         {Michailovich}, Oleg and {Zibulevsky}, Michael and {Bronstein}, Alex},
        title = "{PILOT: Physics-Informed Learned Optimal Trajectories for Accelerated MRI}",
      journal = {arXiv e-prints},
         year = "2019",
archivePrefix = {arXiv},
       eprint = {1909.05773},
}
```

## References
We use the [fastMRI](https://github.com/facebookresearch/fastMRI) as starter template.
We also used [Sigpy](https://github.com/mikgroup/sigpy) as base to our pytorch-nufft implementation.
