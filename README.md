# BayesianAutoencoders
Revisiting Bayesian Autoencoders with MCMC

Autoencoders gained popularity in the deep learning revolution given their ability to compress data and provide dimensionality reduction. Although prominent deep learning methods have been used to enhance autoencoders, the need to provide robust uncertainty quantification remains a challenge. This has been addressed with variational autoencoders so far. Bayesian inference via MCMC methods have faced limitations but recent advances with parallel computing and advanced proposal schemes that incorporate gradients have opened routes less travelled. In this paper, we present Bayesian autoencoders powered MCMC sampling implemented using parallel computing and Langevin gradient proposal scheme. Our proposed Bayesian autoencoder provides similar performance accuracy when compared to related methods from the literature, with the additional feature of robust uncertainty quantification in compressed datasets. This motivates further application of the Bayesian autoencoder framework for other deep learning models.


# Requirements

# Running Parallel Tempering Bayesian Autoencoder

# Datasets

The datasets used in the project can be found here:

1. [Swiss Roll] (https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_swiss_roll.html)
2. [Madelon] (https://archive.ics.uci.edu/ml/datasets/madelon)
3. [Coil 2000] (https://archive.ics.uci.edu/ml/datasets/Insurance+Company+Benchmark+(COIL+2000))

# Acknowledgements
