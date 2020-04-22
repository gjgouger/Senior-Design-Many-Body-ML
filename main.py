from learnClassInit import learn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
hamiltonian = [ "cluster","cluster","cluster"]
learnRate = [ 0.01,0.01, 0.01, 0.01]
learnFunc = [ "momentum","momentum","direct"]
descentMethod = [ "linear descent","SR","SR"]
values = []
numberPlots = len(hamiltonian)
fig, axs = plt.subplots(1, numberPlots, sharex=True,
                        sharey=True, figsize=(20, 8))
for i in range(len(hamiltonian)):
    titleString = ""
    values.append(  
        learn(hamiltonian[i], learnRate[i], learnFunc[i], descentMethod[i]))
    start = time.time()
    values[i].gradDescent()
    end = time.time()
    print("Trial "+str(i)+" took "+str(end-start)+" second to run")

    axs[i].plot(values[i].eStored)
    titleString = (descentMethod[i]+" with "+learnFunc[i]+" descent \nwith learn rate "
                   + str(learnRate[i])+" for hamiltonian "+hamiltonian[0])
    axs[i].set(xlabel='Iteration #', ylabel='Energy')
    if(hamiltonian[i] == "tfim"):
        titleString = titleString+" for g="+str(learn.g)
    if (learnFunc[i] == "momentum"):
        titleString = titleString+"\n with beta "+str(values[i].beta)
    if (learnFunc[i] == "adam"):
        titleString = titleString+"\n with beta1 " + \
            str(values[i].beta1)+" with beta2 "+str(values[i].beta2)
    axs[i].grid()
    axs[i].set(title=titleString)
fig.suptitle("Grad Descent for " +
             str(values[0].n) + " particles", fontsize=14)
name = str(learn.n)+" particles"
for i in range(len(hamiltonian)):
    name = name+" " + \
        hamiltonian[i]+" "+str(learnRate[i])+" " + \
        learnFunc[i]+" "+descentMethod[i]+", "
plt.savefig(name+"2.png")
plt.show()
