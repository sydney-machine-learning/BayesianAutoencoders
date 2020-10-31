import numpy as np
import matplotlib.pyplot as plt
no_chains = int(input("No of chains: "))
no_samples=int(input("Enter no of samples per chain: "))
burnin = 0.53
no_samples_b = int(no_samples-no_samples*burnin)
weight0 =np.zeros((no_chains, no_samples_b))
weight100 =np.zeros((no_chains, no_samples_b))
weight5000 =np.zeros((no_chains, no_samples_b))
weight10000 =np.zeros((no_chains, no_samples_b))
file1 = open("gelman_reuben/rhat.txt","w")




temp = [1.0, 1.640670712015276, 1.1040895136738123, 1.2190136542044754, 1.3459001926323562, 1.4859942891369484, 1.8114473285278132, 2.0]


for i in range(no_chains):

    file_name = 'weight[0]_'+str(temp[i]) + '.txt'
    dat = np.loadtxt(file_name)
    dat = dat[int(burnin*no_samples):]
    weight0[i, :] = dat

    file_name = 'weight[100]_'+str(temp[i]) + '.txt'
    dat = np.loadtxt(file_name)
    dat = dat[int(burnin * no_samples):]
    weight100[i, :] = dat

    file_name = 'weight[5000]_'+str(temp[i]) + '.txt'
    dat = np.loadtxt(file_name)
    dat = dat[int(burnin * no_samples):]
    weight5000[i, :] = dat

    file_name = 'weight[10000]_'+str(temp[i]) + '.txt'
    dat = np.loadtxt(file_name)
    dat = dat[int(burnin * no_samples):]
    weight10000[i, :] = dat


    x1 = np.linspace(0, no_samples_b, num=no_samples_b)

    plt.plot(x1, weight0[i], label='Weight[0]')
    plt.legend(loc='upper right')
    plt.title("Weight[0]_Chain"+str(temp[i]) + " Trace")
    plt.savefig(
        'gelman_reuben/weight[0]_Chain' + str(temp[i])+'_samples.png')
    plt.clf()

    plt.hist(weight0[i], bins=20, color="blue", alpha=0.7)
    plt.ylabel('Frequency')
    plt.xlabel('Parameter Values')
    plt.savefig(
        'gelman_reuben/weight[0]_Chain' + str(temp[i])+'_hist.png')
    plt.clf()

    plt.plot(x1, weight100[i], label='Weight[100]')
    plt.legend(loc='upper right')
    plt.title("Weight[100]_Chain" + str(temp[i]) + " Trace")
    plt.savefig(
        'gelman_reuben/weight[100]_Chain' + str(temp[i]) + '_samples.png')
    plt.clf()

    plt.hist(weight100[i], bins=20, color="blue", alpha=0.7)
    plt.ylabel('Frequency')
    plt.xlabel('Parameter Values')
    plt.savefig(
        'gelman_reuben/weight[100]_Chain' + str(temp[i]) + '_hist.png')
    plt.clf()

    plt.plot(x1, weight5000[i], label='Weight[5000]')
    plt.legend(loc='upper right')
    plt.title("Weight[5000]_Chain" + str(temp[i]) + " Trace")
    plt.savefig(
        'gelman_reuben/weight[5000]_Chain' + str(temp[i]) + '_samples.png')
    plt.clf()

    plt.hist(weight5000[i], bins=20, color="blue", alpha=0.7)
    plt.ylabel('Frequency')
    plt.xlabel('Parameter Values')
    plt.savefig(
        'gelman_reuben/weight[5000]_Chain' + str(temp[i]) + '_hist.png')
    plt.clf()

    plt.plot(x1, weight10000[i], label='Weight[10000]')
    plt.legend(loc='upper right')
    plt.title("Weight[10000]_Chain" + str(temp[i]) + " Trace")
    plt.savefig(
        'gelman_reuben/weight[10000]_Chain' + str(temp[i]) + '_samples.png')
    plt.clf()

    plt.hist(weight10000[i], bins=20, color="blue", alpha=0.7)
    plt.ylabel('Frequency')
    plt.xlabel('Parameter Values')
    plt.savefig(
        'gelman_reuben/weight[10000]_Chain' + str(temp[i]) + '_hist.png')
    plt.clf()




weight0 = weight0.T
weight100 = weight100.T
weight5000 = weight5000.T
weight10000 = weight10000.T

data = np.stack((weight0, weight100, weight5000, weight10000), axis=0)

Nchains, Nsamples, Npars = data.shape

B_on_n = data.mean(axis=1).var(axis=0)      # variance of in-chain means
W = data.var(axis=1).mean(axis=0)           # mean of in-chain variances

#print(B_on_n, ' B_on_n mean')

#print(W, ' W variance ')

# simple version, as in Obsidian
sig2 = (Nsamples/(Nsamples-1))*W + B_on_n
Vhat = sig2 + B_on_n/Nchains
Rhat = Vhat/W

print(Rhat, ' Rhat')
file1.write(str(Rhat))


# advanced version that accounts for ndof
m, n = np.float(Nchains), np.float(Nsamples)
si2 = data.var(axis=1)
xi_bar = data.mean(axis=1)
xi2_bar = data.mean(axis=1)**2
var_si2 = data.var(axis=1).var(axis=0)
allmean = data.mean(axis=1).mean(axis=0)
cov_term1 = np.array([np.cov(si2[:,i], xi2_bar[:,i])[0,1]
                        for i in range(Npars)])
cov_term2 = np.array([-2*allmean[i]*(np.cov(si2[:,i], xi_bar[:,i])[0,1])
                        for i in range(Npars)])
var_Vhat = ( ((n-1)/n)**2 * 1.0/m * var_si2
            +   ((m+1)/m)**2 * 2.0/(m-1) * B_on_n**2
            +   2.0*(m+1)*(n-1)/(m*n**2)
                * n/m * (cov_term1 + cov_term2))
df = 2*Vhat**2 / var_Vhat

#print(df, ' df ')
#print(var_Vhat, ' var_Vhat')
#print "gelman_rubin(): var_Vhat = {}, df = {}".format(var_Vhat, df)


Rhat *= df/(df-2)

print(Rhat, ' Rhat Advanced')
file1.write(str(Rhat))
