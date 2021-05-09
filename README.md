# Final_Project_PPG

This is code for training agents using [Phasic Policy Gradient](https://arxiv.org/abs/2009.04416) [(citation)](#citation) with different clipping methods.

Supported platforms:

- macOS 10.14 (Mojave)
- Ubuntu 16.04

Supported Pythons:

- 3.7 64-bit

## Install

You can get miniconda from https://docs.conda.io/en/latest/miniconda.html if you don't have it, or install the dependencies from [`environment.yml`](environment.yml) manually.

```
git clone https://github.com/SiliGuo/Final_Project_PPG.git
conda env update --name phasic-policy-gradient --file Final_Project_PPG/environment.yml
conda activate phasic-policy-gradient
pip install -e phasic-policy-gradient
```

## Execute

PPG (single network):

```
python -m phasic_policy_gradient.train  --arch detach
```

PPG (single network) with functional clipping method:

```
python -m phasic_policy_gradient.train  --arch detach --clip_method functional
```

PPG (single network) with linearly decaying clipping method:

```
python -m phasic_policy_gradient.train  --arch detach --clip_method decaying
```

PPG (single network, 4 workers):

```
mpiexec -np 4 python -m phasic_policy_gradient.train  --arch detach
```

PPG (single network, 4 workers) with functional clipping method:

```
mpiexec -np 4 python -m phasic_policy_gradient.train  --arch detach --clip_method functional
```

PPG (single network, 4 workers) with linearly decaying clipping method:

```
mpiexec -np 4 python -m phasic_policy_gradient.train  --arch detach --clip_method decaying
```

PPO:

```
python -m phasic_policy_gradient.train --n_epoch_pi 3 --n_epoch_vf 3 --n_aux_epochs 0 --arch shared
```

PPO with functional clipping method:

```
python -m phasic_policy_gradient.train --n_epoch_pi 3 --n_epoch_vf 3 --n_aux_epochs 0 --arch shared --clip_method functional
```

PPO with linearly decaying clipping method:

```
python -m phasic_policy_gradient.train --n_epoch_pi 3 --n_epoch_vf 3 --n_aux_epochs 0 --arch shared --clip_method decaying
```

PPO (4 workers):

```
mpiexec -np 4 python -m phasic_policy_gradient.train --n_epoch_pi 3 --n_epoch_vf 3 --n_aux_epochs 0 --arch shared
```

PPO (4 workers) with functional clipping method:

```
mpiexec -np 4 python -m phasic_policy_gradient.train --n_epoch_pi 3 --n_epoch_vf 3 --n_aux_epochs 0 --arch shared --clip_method functional
```

PPO (4 workers) with linearly decaying clipping method:

```
mpiexec -np 4 python -m phasic_policy_gradient.train --n_epoch_pi 3 --n_epoch_vf 3 --n_aux_epochs 0 --arch shared --clip_method decaying
```

# Citation

Please cite using the following bibtex entry:

```
@article{cobbe2020ppg,
  title={Phasic Policy Gradient},
  author={Cobbe, Karl and Hilton, Jacob and Klimov, Oleg and Schulman, John},
  journal={arXiv preprint arXiv:2009.04416},
  year={2020}
}
```
