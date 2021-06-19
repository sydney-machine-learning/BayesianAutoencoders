# BayesianAutoencoders
Revisiting Bayesian Autoencoders with MCMC

Autoencoders gained popularity in the deep learning revolution given their ability to compress data and provide dimensionality reduction. Although prominent deep learning methods have been used to enhance autoencoders, the need to provide robust uncertainty quantification remains a challenge. This has been addressed with variational autoencoders so far. Bayesian inference via MCMC methods have faced limitations but recent advances with parallel computing and advanced proposal schemes that incorporate gradients have opened routes less travelled. In this paper, we present Bayesian autoencoders powered MCMC sampling implemented using parallel computing and Langevin gradient proposal scheme. Our proposed Bayesian autoencoder provides similar performance accuracy when compared to related methods from the literature, with the additional feature of robust uncertainty quantification in compressed datasets. This motivates further application of the Bayesian autoencoder framework for other deep learning models.

![autoencoder_features](https://user-images.githubusercontent.com/55910983/122639592-588d2f00-d118-11eb-9d58-2cb2044ebef0.png)


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


# Research

We perform preliminary analysis to prove the effectiveness of the bayesian autoencoder approach for dimensionality reduction as compared to a simple (canonical) autoencoder. 

*Simple:*

![Swiss_SA](https://user-images.githubusercontent.com/55910983/122639699-e49f5680-d118-11eb-8ae5-1f5553a0ed3a.png)
 
*Bayesian:*

![Swiss_BA](https://user-images.githubusercontent.com/55910983/122639710-f7b22680-d118-11eb-9a3a-1cc6a5aa05e1.png)

As part of the Bayesian Autoencoder framework, we are able to form a posterior distribution of the parameters as a result of the experiments and we plot the distributuon of some of the randomly selected weights along with the trace of the values of these paramters over the course of the experimentation:

![Madelon_Trace+Post](https://user-images.githubusercontent.com/55910983/122640021-b458b780-d11a-11eb-8270-6a2ab85898e2.png)


To quantify the dimensionality reduction, we calculate the reconstruction accuracy between the original and reconstructed dataset and we trace this value over the course of the experimentation.

![Madelon_Accuracy](https://user-images.githubusercontent.com/55910983/122640130-5ed0da80-d11b-11eb-9e39-bdf1d1249b96.png)


The complete research can be accessed [here](https://arxiv.org/abs/2104.05915)

# Acknowledgements

The team that worked on this project and paper include:-

* Dr. Rohitash Chandra
* Dr. Pavel N. Krivitsky
* Manavendra Maharana
* Mahir Jain





