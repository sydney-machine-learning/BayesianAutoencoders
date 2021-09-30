import numpy as np
import matplotlib.pyplot as plt
no_chains = int(input("No of chains: "))
no_samples=int(input("Enter no of samples per chain: "))
burnin = 0.5
no_samples_b = int(no_samples-no_samples*burnin)
weight0 = np.zeros((no_chains, no_samples_b))
weight100 = np.zeros((no_chains, no_samples_b))
weight5000 = np.zeros((no_chains, no_samples_b))
weight10000 = np.zeros((no_chains, no_samples_b))

weight2000 = np.zeros((no_chains, no_samples_b))
weight3000 = np.zeros((no_chains, no_samples_b))
weight4000 = np.zeros((no_chains, no_samples_b))
weight6000 = np.zeros((no_chains, no_samples_b))
weight7000 = np.zeros((no_chains, no_samples_b))
weight8000 = np.zeros((no_chains, no_samples_b))
weight9000 = np.zeros((no_chains, no_samples_b))
weight11000 = np.zeros((no_chains, no_samples_b))
likelihood_value = np.zeros((no_chains, no_samples))
likelihood_proposal_value = np.zeros((no_chains, no_samples))
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

    file_name = 'weight[2000]_'+str(temp[i]) + '.txt'
    dat = np.loadtxt(file_name)
    dat = dat[int(burnin*no_samples):]
    weight2000[i, :] = dat

    file_name = 'weight[3000]_'+str(temp[i]) + '.txt'
    dat = np.loadtxt(file_name)
    dat = dat[int(burnin * no_samples):]
    weight3000[i, :] = dat

    file_name = 'weight[4000]_'+str(temp[i]) + '.txt'
    dat = np.loadtxt(file_name)
    dat = dat[int(burnin * no_samples):]
    weight4000[i, :] = dat

    file_name = 'weight[6000]_'+str(temp[i]) + '.txt'
    dat = np.loadtxt(file_name)
    dat = dat[int(burnin * no_samples):]
    weight6000[i, :] = dat

    file_name = 'weight[7000]_'+str(temp[i]) + '.txt'
    dat = np.loadtxt(file_name)
    dat = dat[int(burnin * no_samples):]
    weight7000[i, :] = dat

    file_name = 'weight[8000]_'+str(temp[i]) + '.txt'
    dat = np.loadtxt(file_name)
    dat = dat[int(burnin * no_samples):]
    weight8000[i, :] = dat

    file_name = 'weight[9000]_'+str(temp[i]) + '.txt'
    dat = np.loadtxt(file_name)
    dat = dat[int(burnin * no_samples):]
    weight9000[i, :] = dat

    file_name = 'weight[11000]_'+str(temp[i]) + '.txt'
    dat = np.loadtxt(file_name)
    dat = dat[int(burnin * no_samples):]
    weight11000[i, :] = dat

    file_name = 'likelihood_value_'+str(temp[i]) + '.txt'
    likelihood_value[i, :] = np.loadtxt(file_name)

    file_name = 'likelihood_value_proposal'+str(temp[i]) + '.txt'
    likelihood_proposal_value[i, :] = np.loadtxt(file_name)



    file_name = 'acc_test_chain_1.0.txt'
    acc_test = np.loadtxt(file_name)

    file_name = 'acc_train_chain_1.0.txt'
    acc_train = np.loadtxt(file_name)

    x1 = np.linspace(0, no_samples_b, num=no_samples_b)

    plt.plot(x1, weight0[i], label='Weight[0]')
    plt.legend(loc='upper right')
    #plt.title("Weight[0]_Chain"+str(temp[i]) + " Trace")
    #plt.ylim(-1,1)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(
        'gelman_reuben/weight[0]_Chain' + str(temp[i])+'_samples.png')
    plt.clf()


    plt.hist(weight0[i], bins=20, color="blue", alpha=0.7)
    #plt.title('Posterior Distribution')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('Frequency')
    plt.xlabel('Parameter Values')
    plt.savefig(
        'gelman_reuben/weight[0]_Chain' + str(temp[i])+'_hist.png')
    plt.clf()

    plt.plot(x1, weight100[i], label='Weight[100]')
    plt.legend(loc='upper right')
    #plt.title("Weight[100]_Chain" + str(temp[i]) + " Trace")
    #plt.ylim(-1,1)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(
        'gelman_reuben/weight[100]_Chain' + str(temp[i]) + '_samples.png')
    plt.clf()

    plt.hist(weight100[i], bins=20, color="blue", alpha=0.7)
    #plt.title('Posterior Distribution')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('Frequency')
    plt.xlabel('Parameter Values')
    plt.savefig(
        'gelman_reuben/weight[100]_Chain' + str(temp[i]) + '_hist.png')
    plt.clf()

    plt.plot(x1, weight5000[i], label='Weight[5000]')
    plt.legend(loc='upper right')
    #plt.title("Weight[5000]_Chain" + str(temp[i]) + " Trace")
    #plt.ylim(-1,1)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(
        'gelman_reuben/weight[5000]_Chain' + str(temp[i]) + '_samples.png')
    plt.clf()

    plt.hist(weight5000[i], bins=20, color="blue", alpha=0.7)
    #plt.title('Posterior Distribution')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('Frequency')
    plt.xlabel('Parameter Values')
    plt.savefig(
        'gelman_reuben/weight[5000]_Chain' + str(temp[i]) + '_hist.png')
    plt.clf()

    plt.plot(x1, weight10000[i], label='Weight[10000]')
    plt.legend(loc='upper right')
    #plt.title("Weight[10000]_Chain" + str(temp[i]) + " Trace")
    #plt.ylim(-1,1)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(
        'gelman_reuben/weight[10000]_Chain' + str(temp[i]) + '_samples.png')
    plt.clf()

    plt.hist(weight10000[i], bins=20, color="blue", alpha=0.7)
    #plt.title('Posterior Distribution')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('Frequency')
    plt.xlabel('Parameter Values')
    plt.savefig(
        'gelman_reuben/weight[10000]_Chain' + str(temp[i]) + '_hist.png')
    plt.clf()

    plt.hist(weight10000[i], bins=20, color="blue", alpha=0.7)
    #plt.title('Posterior Distribution')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('Frequency')
    plt.xlabel('Parameter Values')
    plt.savefig(
        'gelman_reuben/weight[10000]_Chain' + str(temp[i]) + '_hist.png')
    plt.clf()

    plt.hist(weight2000[i], bins=20, color="blue", alpha=0.7)
    #plt.title('Posterior Distribution')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('Frequency')
    plt.xlabel('Parameter Values')
    plt.savefig(
        'gelman_reuben/weight[2000]_Chain' + str(temp[i]) + '_hist.png')
    plt.clf()

    plt.hist(weight3000[i], bins=20, color="blue", alpha=0.7)
    #plt.title('Posterior Distribution')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('Frequency')
    plt.xlabel('Parameter Values')
    plt.savefig(
        'gelman_reuben/weight[3000]_Chain' + str(temp[i]) + '_hist.png')
    plt.clf()

    plt.hist(weight4000[i], bins=20, color="blue", alpha=0.7)
    #plt.title('Posterior Distribution')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('Frequency')
    plt.xlabel('Parameter Values')
    plt.savefig(
        'gelman_reuben/weight[4000]_Chain' + str(temp[i]) + '_hist.png')
    plt.clf()

    plt.hist(weight6000[i], bins=20, color="blue", alpha=0.7)
    #plt.title('Posterior Distribution')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('Frequency')
    plt.xlabel('Parameter Values')
    plt.savefig(
        'gelman_reuben/weight[6000]_Chain' + str(temp[i]) + '_hist.png')
    plt.clf()

    plt.hist(weight7000[i], bins=20, color="blue", alpha=0.7)
    #plt.title('Posterior Distribution')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('Frequency')
    plt.xlabel('Parameter Values')
    plt.savefig(
        'gelman_reuben/weight[7000]_Chain' + str(temp[i]) + '_hist.png')
    plt.clf()

    plt.hist(weight8000[i], bins=20, color="blue", alpha=0.7)
    #plt.title('Posterior Distribution')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('Frequency')
    plt.xlabel('Parameter Values')
    plt.savefig(
        'gelman_reuben/weight[8000]_Chain' + str(temp[i]) + '_hist.png')
    plt.clf()

    plt.hist(weight9000[i], bins=20, color="blue", alpha=0.7)
    #plt.title('Posterior Distribution')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('Frequency')
    plt.xlabel('Parameter Values')
    plt.savefig(
        'gelman_reuben/weight[9000]_Chain' + str(temp[i]) + '_hist.png')
    plt.clf()

    plt.hist(weight11000[i], bins=20, color="blue", alpha=0.7)
    #plt.title('Posterior Distribution')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('Frequency')
    plt.xlabel('Parameter Values')
    plt.savefig(
        'gelman_reuben/weight[11000]_Chain' + str(temp[i]) + '_hist.png')
    plt.clf()

x1 = np.linspace(0, no_samples_b, num=no_samples_b)
x2 = np.linspace(0, no_samples, num=no_samples)

plt.plot(x1, weight0[0], label=str(temp[0]))
plt.plot(x1, weight0[1], label=str(temp[1]))
plt.plot(x1, weight0[2], label=str(temp[2]))
plt.plot(x1, weight0[3], label=str(temp[3]))
plt.plot(x1, weight0[4], label=str(temp[4]))
plt.plot(x1, weight0[5], label=str(temp[5]))
plt.plot(x1, weight0[6], label=str(temp[6]))
plt.plot(x1, weight0[7], label=str(temp[7]))
plt.legend(loc='lower right')
#plt.title("Trace Plot")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Samples")
plt.ylabel("Parameter Values")
plt.ylim(-4, 4)
plt.tight_layout()
plt.savefig(
'gelman_reuben/weight[0]_Chain' +'_samples.png')
plt.clf()

plt.plot(x1, weight100[0], label=str(temp[0]))
#plt.plot(x1, weight100[1], label=str(temp[1]))
#plt.plot(x1, weight100[2], label=str(temp[2]))
#plt.plot(x1, weight100[3], label=str(temp[3]))
#plt.plot(x1, weight100[4], label=str(temp[4]))
#plt.plot(x1, weight100[5], label=str(temp[5]))
#plt.plot(x1, weight100[6], label=str(temp[6]))
plt.plot(x1, weight100[7], label=str(temp[7]))
plt.legend(loc='lower right')
#plt.title("Trace Plot")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Samples")
plt.ylabel("Parameter Values")
plt.ylim(-4, 4)
plt.tight_layout()
plt.savefig(
'gelman_reuben/weight[100]_Chain' +'_samples.png')
plt.clf()

plt.plot(x1, weight5000[0], label=str(temp[0]))
#plt.plot(x1, weight5000[1], label=str(temp[1]))
#plt.plot(x1, weight5000[2], label=str(temp[2]))
#plt.plot(x1, weight5000[3], label=str(temp[3]))
#plt.plot(x1, weight5000[4], label=str(temp[4]))
#plt.plot(x1, weight5000[5], label=str(temp[5]))
#plt.plot(x1, weight5000[6], label=str(temp[6]))
plt.plot(x1, weight5000[7], label=str(temp[7]))
plt.legend(loc='lower right')
#plt.title("Trace Plot")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Samples")
plt.ylabel("Parameter Values")
plt.ylim(-4, 4)
plt.tight_layout()
plt.savefig(
'gelman_reuben/weight[5000]_Chain' +'_samples.png')
plt.clf()

plt.plot(x1, weight10000[0], label=str(temp[0]))
#plt.plot(x1, weight10000[1], label='Weight[10000]'+str(temp[1]))
#plt.plot(x1, weight10000[2], label='Weight[10000]'+str(temp[2]))
#plt.plot(x1, weight10000[3], label='Weight[10000]'+str(temp[3]))
#plt.plot(x1, weight10000[4], label='Weight[10000]'+str(temp[4]))
#plt.plot(x1, weight10000[5], label='Weight[10000]'+str(temp[5]))
#plt.plot(x1, weight10000[6], label='Weight[10000]'+str(temp[6]))
plt.plot(x1, weight10000[7], label=str(temp[7]))
plt.legend(loc='lower right')
#plt.title("Trace Plot")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Samples")
plt.ylabel("Parameter Values")
plt.ylim(-4, 4)
plt.tight_layout()
plt.savefig(
'gelman_reuben/weight[10000]_Chain' +'_samples.png')
plt.clf()

plt.plot(x1, weight0[0], label=str(temp[0]))
#plt.plot(x1, weight0[1], label=str(temp[1]))
#plt.plot(x1, weight0[2], label=str(temp[2]))
#plt.plot(x1, weight0[3], label=str(temp[3]))
#plt.plot(x1, weight0[4], label=str(temp[4]))
#plt.plot(x1, weight0[5], label=str(temp[5]))
#plt.plot(x1, weight0[6], label=str(temp[6]))
plt.plot(x1, weight0[7], label=str(temp[7]))
plt.legend(loc='lower right')
#plt.title("Trace Plot")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Samples")
plt.ylabel("Parameter Values")
plt.ylim(-4, 4)
plt.tight_layout()
plt.savefig(
'gelman_reuben/weight[0]_Chain' +'_samples.png')
plt.clf()

plt.plot(x1, weight100[0], label=str(temp[0]))
#plt.plot(x1, weight100[1], label=str(temp[1]))
#plt.plot(x1, weight100[2], label=str(temp[2]))
#plt.plot(x1, weight100[3], label=str(temp[3]))
#plt.plot(x1, weight100[4], label=str(temp[4]))
#plt.plot(x1, weight100[5], label=str(temp[5]))
#plt.plot(x1, weight100[6], label=str(temp[6]))
plt.plot(x1, weight100[7], label=str(temp[7]))
plt.legend(loc='lower right')
#plt.title("Trace Plot")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Samples")
plt.ylabel("Parameter Values")
plt.ylim(-4, 4)
plt.tight_layout()
plt.savefig(
'gelman_reuben/weight[100]_Chain' +'_samples.png')
plt.clf()

plt.plot(x1, weight5000[0], label=str(temp[0]))
#plt.plot(x1, weight5000[1], label=str(temp[1]))
#plt.plot(x1, weight5000[2], label=str(temp[2]))
#plt.plot(x1, weight5000[3], label=str(temp[3]))
#plt.plot(x1, weight5000[4], label=str(temp[4]))
#plt.plot(x1, weight5000[5], label=str(temp[5]))
#plt.plot(x1, weight5000[6], label=str(temp[6]))
plt.plot(x1, weight5000[7], label=str(temp[7]))
plt.legend(loc='lower right')
#plt.title("Trace Plot")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Samples")
plt.ylabel("Parameter Values")
plt.ylim(-4, 4)
plt.tight_layout()
plt.savefig(
'gelman_reuben/weight[5000]_Chain' +'_samples.png')
plt.clf()

plt.plot(x1, weight10000[0], label=str(temp[0]))
#plt.plot(x1, weight10000[1], label='Weight[10000]'+str(temp[1]))
#plt.plot(x1, weight10000[2], label='Weight[10000]'+str(temp[2]))
#plt.plot(x1, weight10000[3], label='Weight[10000]'+str(temp[3]))
#plt.plot(x1, weight10000[4], label='Weight[10000]'+str(temp[4]))
#plt.plot(x1, weight10000[5], label='Weight[10000]'+str(temp[5]))
#plt.plot(x1, weight10000[6], label='Weight[10000]'+str(temp[6]))
plt.plot(x1, weight10000[7], label=str(temp[7]))
plt.legend(loc='lower right')
#plt.title("Trace Plot")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Samples")
plt.ylabel("Parameter Values")
plt.ylim(-4, 4)
plt.tight_layout()
plt.savefig(
'gelman_reuben/weight[10000]_Chain' +'_samples.png')
plt.clf()

plt.plot(x1, weight2000[0], label=str(temp[0]))
#plt.plot(x1, weight2000[1], label='Weight[2000]'+str(temp[1]))
#plt.plot(x1, weight2000[2], label='Weight[2000]'+str(temp[2]))
#plt.plot(x1, weight2000[3], label='Weight[2000]'+str(temp[3]))
#plt.plot(x1, weight2000[4], label='Weight[2000]'+str(temp[4]))
#plt.plot(x1, weight2000[5], label='Weight[2000]'+str(temp[5]))
#plt.plot(x1, weight2000[6], label='Weight[2000]'+str(temp[6]))
plt.plot(x1, weight2000[7], label=str(temp[7]))
plt.legend(loc='lower right')
#plt.title("Trace Plot")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Samples")
plt.ylabel("Parameter Values")
plt.ylim(-4, 4)
plt.tight_layout()
plt.savefig(
'gelman_reuben/weight[2000]_Chain' +'_samples.png')
plt.clf()

plt.plot(x1, weight3000[0], label=str(temp[0]))
#plt.plot(x1, weight3000[1], label=str(temp[1]))
#plt.plot(x1, weight3000[2], label=str(temp[2]))
#plt.plot(x1, weight3000[3], label=str(temp[3]))
#plt.plot(x1, weight3000[4], label=str(temp[4]))
#plt.plot(x1, weight3000[5], label=str(temp[5]))
#plt.plot(x1, weight3000[6], label=str(temp[6]))
plt.plot(x1, weight3000[7], label=str(temp[7]))
plt.legend(loc='lower right')
#plt.title("Trace Plot")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Samples")
plt.ylabel("Parameter Values")
plt.ylim(-4, 4)
plt.tight_layout()
plt.savefig(
'gelman_reuben/weight[3000]_Chain' +'_samples.png')
plt.clf()

plt.plot(x1, weight4000[0], label=str(temp[0]))
#plt.plot(x1, weight4000[1], label=str(temp[1]))
#plt.plot(x1, weight4000[2], label=str(temp[2]))
#plt.plot(x1, weight4000[3], label=str(temp[3]))
#plt.plot(x1, weight4000[4], label=str(temp[4]))
#plt.plot(x1, weight4000[5], label=str(temp[5]))
#plt.plot(x1, weight4000[6], label=str(temp[6]))
plt.plot(x1, weight4000[7], label=str(temp[7]))
plt.legend(loc='lower right')
#plt.title("Trace Plot")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Samples")
plt.ylabel("Parameter Values")
plt.ylim(-4, 4)
plt.tight_layout()
plt.savefig(
'gelman_reuben/weight[4000]_Chain' +'_samples.png')
plt.clf()

plt.plot(x1, weight6000[0], label=str(temp[0]))
#plt.plot(x1, weight6000[1], label='Weight[6000]'+str(temp[1]))
#plt.plot(x1, weight6000[2], label='Weight[6000]'+str(temp[2]))
#plt.plot(x1, weight6000[3], label='Weight[6000]'+str(temp[3]))
#plt.plot(x1, weight6000[4], label='Weight[6000]'+str(temp[4]))
#plt.plot(x1, weight6000[5], label='Weight[6000]'+str(temp[5]))
#plt.plot(x1, weight6000[6], label='Weight[6000]'+str(temp[6]))
plt.plot(x1, weight6000[7], label=str(temp[7]))
plt.legend(loc='lower right')
#plt.title("Trace Plot")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Samples")
plt.ylabel("Parameter Values")
plt.ylim(-4, 4)
plt.tight_layout()
plt.savefig(
'gelman_reuben/weight[6000]_Chain' +'_samples.png')
plt.clf()

plt.plot(x1, weight7000[0], label=str(temp[0]))
#plt.plot(x1, weight7000[1], label=str(temp[1]))
#plt.plot(x1, weight7000[2], label=str(temp[2]))
#plt.plot(x1, weight7000[3], label=str(temp[3]))
#plt.plot(x1, weight7000[4], label=str(temp[4]))
#plt.plot(x1, weight7000[5], label=str(temp[5]))
#plt.plot(x1, weight7000[6], label=str(temp[6]))
plt.plot(x1, weight7000[7], label=str(temp[7]))
plt.legend(loc='lower right')
#plt.title("Trace Plot")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Samples")
plt.ylabel("Parameter Values")
plt.ylim(-4, 4)
plt.tight_layout()
plt.savefig(
'gelman_reuben/weight[7000]_Chain' +'_samples.png')
plt.clf()

plt.plot(x1, weight8000[0], label=str(temp[0]))
#plt.plot(x1, weight8000[1], label=str(temp[1]))
#plt.plot(x1, weight8000[2], label=str(temp[2]))
#plt.plot(x1, weight8000[3], label=str(temp[3]))
#plt.plot(x1, weight8000[4], label=str(temp[4]))
#plt.plot(x1, weight8000[5], label=str(temp[5]))
#plt.plot(x1, weight8000[6], label=str(temp[6]))
plt.plot(x1, weight8000[7], label=str(temp[7]))
plt.legend(loc='lower right')
#plt.title("Trace Plot")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Samples")
plt.ylabel("Parameter Values")
plt.ylim(-4, 4)
plt.tight_layout()
plt.savefig(
'gelman_reuben/weight[8000]_Chain' +'_samples.png')
plt.clf()

plt.plot(x1, weight9000[0], label=str(temp[0]))
#plt.plot(x1, weight9000[1], label='Weight[9000]'+str(temp[1]))
#plt.plot(x1, weight9000[2], label='Weight[9000]'+str(temp[2]))
#plt.plot(x1, weight9000[3], label='Weight[9000]'+str(temp[3]))
#plt.plot(x1, weight9000[4], label='Weight[9000]'+str(temp[4]))
#plt.plot(x1, weight9000[5], label='Weight[9000]'+str(temp[5]))
#plt.plot(x1, weight9000[6], label='Weight[9000]'+str(temp[6]))
plt.plot(x1, weight9000[7], label= str(temp[7]))
plt.legend(loc='lower right')
#plt.title("Trace Plot")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Samples")
plt.ylabel("Parameter Values")
plt.ylim(-4, 4)
plt.tight_layout()
plt.savefig(
'gelman_reuben/weight[9000]_Chain' +'_samples.png')
plt.clf()

plt.plot(x1, weight11000[0], label=str(temp[0]))
#plt.plot(x1, weight11000[1], label='Weight[11000]'+str(temp[1]))
#plt.plot(x1, weight11000[2], label='Weight[11000]'+str(temp[2]))
#plt.plot(x1, weight11000[3], label='Weight[11000]'+str(temp[3]))
#plt.plot(x1, weight11000[4], label='Weight[11000]'+str(temp[4]))
#plt.plot(x1, weight11000[5], label='Weight[11000]'+str(temp[5]))
#plt.plot(x1, weight11000[6], label='Weight[11000]'+str(temp[6]))
plt.plot(x1, weight11000[7], label=str(temp[7]))
plt.legend(loc='lower right')
#plt.title("Trace Plot")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Samples")
plt.ylabel("Parameter Values")
plt.ylim(-4, 4)
plt.tight_layout()
plt.savefig(
'gelman_reuben/weight[11000]_Chain' +'_samples.png')
plt.clf()

plt.plot(x2, likelihood_value[0], label='Log Likelihood '+ str(temp[0]))
#plt.plot(x2, likelihood_value[1], label='Weight[11000]'+str(temp[1]))
#plt.plot(x2, likelihood_value[2], label='Weight[11000]'+str(temp[2]))
#plt.plot(x2, likelihood_value[3], label='Weight[11000]'+str(temp[3]))
plt.plot(x2, likelihood_value[4], label='Log Likelihood'+str(temp[4]))
#plt.plot(x2, likelihood_value[5], label='Weight[11000]'+str(temp[5]))
#plt.plot(x2, likelihood_value[6], label='Weight[11000]'+str(temp[6]))
plt.plot(x2, likelihood_value[7], label='Log Likelihood '+ str(temp[7]))
plt.legend(loc='lower right')
#plt.title("Likelihood Function Trace")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Samples")
plt.ylabel("Log Likelihood Value")
#plt.ylim(-4, 4)
plt.tight_layout()
plt.savefig(
'gelman_reuben/likelihood_value' +'_samples.png')
plt.clf()

plt.plot(x2, likelihood_proposal_value[0], label='Log Likelihood '+ str(temp[0]))
#plt.plot(x2, likelihood_proposal_value[1], label='Weight[11000]'+str(temp[1]))
#plt.plot(x2, likelihood_value[2], label='Weight[11000]'+str(temp[2]))
#plt.plot(x2, likelihood_value[3], label='Weight[11000]'+str(temp[3]))
#plt.plot(x2, likelihood_value[4], label='Weight[11000]'+str(temp[4]))
#plt.plot(x2, likelihood_value[5], label='Weight[11000]'+str(temp[5]))
#plt.plot(x2, likelihood_value[6], label='Weight[11000]'+str(temp[6]))
plt.plot(x2, likelihood_proposal_value[7], label='Log Likelihood '+ str(temp[7]))
plt.legend(loc='lower right')
#plt.title("Proposed Likelihood Values")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Samples")
plt.ylabel("Log Likelihood Value")
#plt.ylim(-4, 4)
plt.tight_layout()
plt.savefig(
'gelman_reuben/likelihood_proposal_value' +'_samples.png')
plt.clf()

plt.plot(x2, likelihood_proposal_value[0], label='Log Likelihood_Proposal ')
plt.plot(x2, likelihood_value[0], label='Log Likelihood')
plt.legend(loc='lower right')
#plt.title("Accepted vs Proposal")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Samples")
plt.ylabel("Log Likelihood Value")
#plt.ylim(-4, 4)
plt.tight_layout()
plt.savefig(
'gelman_reuben/likelihood_avsp' +'_samples.png')
plt.clf()




color = 'tab:red'
plt.plot(x2, acc_train, label="Train", color=color)
color = 'tab:blue'
plt.plot(x2, acc_test, label="Test", color=color)
plt.xlabel('Samples')
plt.ylabel('Reconstruction Accuracy')
#plt.title('Accuracy Value')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend()
plt.savefig('gelman_reuben/accuracy_value' +'_samples.png')
plt.clf()





weight0 = weight0.T
weight100 = weight100.T
weight5000 = weight5000.T
weight10000 = weight10000.T
weight3000 = weight3000.T
weight2000 = weight2000.T
weight4000 = weight4000.T
weight6000 = weight6000.T
weight7000 = weight7000.T
weight8000 = weight8000.T
weight9000 = weight9000.T
weight11000 = weight11000.T


data = np.stack((weight0, weight100, weight2000, weight3000, weight4000, weight5000, weight6000, weight7000, weight8000, weight9000, weight10000, weight11000), axis=2)
#data = np.array([weight0,weight100,weight5000,weight10000])
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
