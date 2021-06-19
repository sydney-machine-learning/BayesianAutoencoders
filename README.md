# BayesianAutoencoders
Revisiting Bayesian Autoencoders with MCMC

Autoencoders gained popularity in the deep learning revolution given their ability to compress data and provide dimensionality reduction. Although prominent deep learning methods have been used to enhance autoencoders, the need to provide robust uncertainty quantification remains a challenge. This has been addressed with variational autoencoders so far. Bayesian inference via MCMC methods have faced limitations but recent advances with parallel computing and advanced proposal schemes that incorporate gradients have opened routes less travelled. In this paper, we present Bayesian autoencoders powered MCMC sampling implemented using parallel computing and Langevin gradient proposal scheme. Our proposed Bayesian autoencoder provides similar performance accuracy when compared to related methods from the literature, with the additional feature of robust uncertainty quantification in compressed datasets. This motivates further application of the Bayesian autoencoder framework for other deep learning models.


# Requirements
This project was tested on python 3.7 and the required packages can be installed using the provided requirements.txt:

```python
    pip install -r requirements.txt
```

# Running Parallel Tempering Bayesian Autoencoder
```python
    cd Bayesian
    python Parallel_Tempering_Tabular.py
```
After running these lines, you will be prompted to enter the following values:- 
* Number of Samples- These are the total number of samples for which the code will run and this will be distributed across 8 Chains to take advantage of parallel computing (Recommended Number of Samples: >48000)
* Dataset- Since we run experiments using any of the three datasets, you must choose the dataset you want to use for experimentation.

It is also recommended to use tools like [Google Colab](https://colab.research.google.com/) to run the code since these are optimized to take advantage of Parallel Computing.


# Datasets

The datasets used in the project can be found here:

1. [Swiss Roll](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_swiss_roll.html)
2. [Madelon](https://archive.ics.uci.edu/ml/datasets/madelon)
3. [Coil 2000](https://archive.ics.uci.edu/ml/datasets/Insurance+Company+Benchmark+(COIL+2000))


# Preliminary Analysis

# Acknowledgements

The team that worked on this project and paper include:-

* Dr. Rohitash Chandra
* Dr. Pavel N. Krivitsky
* Manavendra Maharana
* Mahir Jain

