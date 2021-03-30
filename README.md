# RRAM
A MATLAB solver for low-rank matrix completion based on a **R**iemannian **R**ank-**A**daptive **M**ethod

## How to run

1. Run ``startup.m`` and ``Install_mex.m`` first.
```
run startup.m
run Install_mex.m
```
2. Demo scripts are listed in ``./Examples`` including
  + ``test_fixed_rank.m``
  + ``test_rank_increase.m``
  + ``test_rank_reduction.m``

## References
[Bin Gao](https://www.gaobin.cc/) [P.-A. Absil](https://sites.uclouvain.be/absil/)
+ [A Riemannian rank-adaptive method for low-rank matrix completion](https://arxiv.org/abs/2103.14768)

## Authors
+ [Bin Gao](https://www.gaobin.cc/) (UCLouvain, Belgium)

## Copyright
Copyright (C) 2021, Bin Gao, P.-A. Absil

This package is based on a third-party public code. That copyright is reserved by [Bart Vandereycken](https://www.unige.ch/math/vandereycken/matrix_completion.html).

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/)
