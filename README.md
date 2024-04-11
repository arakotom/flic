## This is the code repo for the paper 

Personalised Federated Learning On Heterogeneous Feature Spaces by A. Rakotomamonjy et al.

The paper is available at [https://openreview.net/pdf?id=uCZJaqJchs]


This repo should allow you to reproduce the results on the BCI datasets and on FEMNIST.



### Structure of the repository

The code is heavily based on the FedRep code of Collins et al. which unfortunately is not maintained anymore on Github. 

The code is structured as follows:

* script_bci.py allows to launch all the experiments with the different algorithms 
* create_bci_dataset.py allows to create the data (which are already in the repo). You will need to install the moabb library.
* script_feMNIST.py allows to launch all the experiments with the different algorithms on the FEMNIST dataset.  the FE-MNIST dataset is already in the repo but they can be built using the LEAF repo.

If you use this code for your research, you can cite our paper:

```
@inproceedings{rakoto2024fedwad,
  title={Federated Wassertein Distance},
  author={Rakotomamonjy, Alain and Nadjahi, Kimia and Ralaivola, Liva},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```

