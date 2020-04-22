# Make intial Values
import numpy as np
from cmath import pi
from scipy.sparse import spdiags
import copy
# Cluster
# from calculateWave import calculateWave


class learn:
    n = 15
    m = 15
    burn = 400
    walks = 1
    metroLoops = 2000
    iterations = 1200
    g = 1 
    pbc = True
    exact = False
    epsilon = 1e-8
    # Momentum variable
    beta = 0.3
    # Adam
    beta1 = .15
    beta2 = .01

    def __init__(self, hamiltonian, learnRate, learnFunc, descentMethod):

        string2FuncDesc = {
            "linear descent": self.calculateGradientsLinear,
            "SR": self.calculateGradientsSGD
        }
        string2FuncHamil = {
            "cluster": self.cluster,
            "tfim": self.tfim
        }
        string2FuncLearn = {
            "direct": self.directLearn,
            "momentum": self.momentumLearn,
            "adaGrad": self.adaGradLearn,
            "adam": self.adamLearn
        }

        self.learnRate = learnRate
        self.makeIntialValues()
        self.E = 0
        self.gradA = None
        self.gradB = None
        self.gradW = None
        self.prevGradA = np.zeros([learn.n, 1])
        self.prevGradB = np.zeros([learn.m, 1])
        self.prevGradW = np.zeros([learn.n, learn.m])
        self.psi = None
        self.otan = None
        self.eStored = []
        self.descentMethod = string2FuncDesc[descentMethod]
        self.learnFunc = string2FuncLearn[learnFunc]
        self.hamiltonian = string2FuncHamil[hamiltonian]
        # adam
        self.gradAv = None
        self.gradBv = None
        self.gradWv = None
        self.prevGradAv = np.zeros([learn.n, 1])
        self.prevGradBv = np.zeros([learn.m, 1])
        self.prevGradWv = np.zeros([learn.n, learn.m])

    def directLearn(self):
        self.a_vec = ((self.a_vec) - (self.gradA) * self.learnRate)
        self.b_vec = ((self.b_vec) - (self.gradB) * self.learnRate)
        self.w_mat = ((self.w_mat) - (self.gradW) * self.learnRate)

    def momentumLearn(self):
        self.gradA = learn.beta*self.prevGradA+(1-learn.beta)*self.gradA
        self.gradB = learn.beta*self.prevGradB+(1-learn.beta)*self.gradB
        self.gradW = learn.beta*self.prevGradW+(1-learn.beta)*self.gradW

        self.prevGradA = self.gradA
        self.prevGradB = self.gradB
        self.prevGradW = self.gradW

        self.a_vec = ((self.a_vec) - self.gradA*self.learnRate)
        self.b_vec = ((self.b_vec) - self.gradB*self.learnRate)
        self.w_mat = ((self.w_mat) - self.gradW*self.learnRate)

    def adaGradLearn(self):
        # adagrad variable

        self.prevGradA = self.prevGradA+np.square(self.gradA)
        self.prevGradB = self.prevGradB+np.square(self.gradB)
        self.prevGradW = self.prevGradW+np.square(self.gradW)

        learnRateA = np.divide(
            self.learnRate, (np.sqrt(self.prevGradA+learn.epsilon)))
        learnRateB = np.divide(
            self.learnRate, (np.sqrt(self.prevGradB+learn.epsilon)))
        learnRateW = np.divide(
            self.learnRate, (np.sqrt(self.prevGradW+learn.epsilon)))

        self.a_vec = ((self.a_vec) - (self.gradA)*learnRateA)
        self.b_vec = ((self.b_vec) - (self.gradB)*learnRateB)
        self.w_mat = ((self.w_mat) - (self.gradW)*learnRateW)

    def adamLearn(self):
        # adagrad variable

        self.gradAv = learn.beta2*self.prevGradAv + \
            (1-learn.beta2)*np.square(self.gradA)
        self.gradBv = learn.beta2*self.prevGradBv + \
            (1-learn.beta2)*np.square(self.gradB)
        self.gradWv = learn.beta2*self.prevGradWv + \
            (1-learn.beta2)*np.square(self.gradW)

        self.gradA = learn.beta1*self.prevGradA+(1-learn.beta1)*self.gradA
        self.gradB = learn.beta1*self.prevGradB+(1-learn.beta1)*self.gradB
        self.gradW = learn.beta1*self.prevGradW+(1-learn.beta1)*self.gradW

        self.prevGradA = self.gradA
        self.prevGradB = self.gradB
        self.prevGradW = self.gradW

        self.prevGradAv = self.gradAv
        self.prevGradBv = self.gradBv
        self.prevGradWv = self.gradWv

        self.gradA = self.gradA/(1-self.beta1)
        self.gradB = self.gradB/(1-self.beta1)
        self.gradW = self.gradW/(1-self.beta1)

        self.gradAv = self.gradAv/(1-self.beta2)
        self.gradBv = self.gradBv/(1-self.beta2)
        self.gradWv = self.gradWv/(1-self.beta2)

        learnRateA = np.divide(
            self.learnRate, (np.sqrt(self.gradAv)+learn.epsilon))
        learnRateB = np.divide(
            self.learnRate, (np.sqrt(self.gradBv)+learn.epsilon))
        learnRateW = np.divide(
            self.learnRate, (np.sqrt(self.gradWv)+learn.epsilon))

        self.a_vec = ((self.a_vec) - (self.gradA)*learnRateA)
        self.b_vec = ((self.b_vec) - (self.gradB)*learnRateB)
        self.w_mat = ((self.w_mat) - (self.gradW)*learnRateW)

    def gradDescent(self):

        for _ in range(learn.iterations):

            self.descentMethod()

            self.eStored.append(self.E[0][0])  # add E to array

            self.learnFunc()

    def cluster(self, s, oldWave):
        '''
        s: Current Spin State

        a_vec: a_vec paramter

        b_vec: b_vec parameter

        w_mat: w_mat parameter 

        g: Dummy variable so that the hamiltonians can easily be swapped out

        oldWave: Previous value of psi

        pbc: Period Boundary Condition

        Output: Energy 

        Calculates the energy for the cluster hamiltonian with the given inputs
        '''
        #  Find local energy at each spin site and average
        energy = 0
        for j in range(learn.n):
            if j == 0:
                if learn.pbc == True:  # check for periodic boundary conditions
                    factor = s[learn.n-1]*s[j+1]
                else:
                    factor = 0
            elif j == learn.n-1:
                s[j-1] = -1*s[j-1]
                if learn.pbc == True:
                    factor = s[j-1]*s[0]
                else:
                    factor = 0
            else:
                s[j-1] = -1*s[j-1]  # reverse flip done by previous iteration
                factor = s[j-1]*s[j+1]  # calculate sigma z factor
            s[j] = -1*s[j]  # flip spin at current site
            self.calculateWave(s)  # find new wavefunction
            newWave = self.psi
            # calculate sigma x, multiply by sigma z factor and add to energy
            energy = energy - factor*(np.conj(newWave/oldWave))
        return energy

    def tfim(self, s, oldWave):
        '''
        s: Current Spin State

        a_vec: a_vec paramter

        b_vec: b_vec parameter

        w_mat: w_mat parameter 

        g: Dummy variable so that the hamiltonians can easily be swapped out

        oldWave: Previous value of psi

        pbc: Period Boundary Condition

        Output: Energy 

        Calculates the energy for the tfim hamiltonian with the given inputs
        '''
        # Transfer Field Ising Model
        xterm = 0
        zterm = 0

        # Find local energy at each spin site and average
        for j in range(learn.n):
            if j == 0:
                factor = s[0]*s[1]
            elif j == learn.n-1:
                s[j-1] = -1*s[j-1]
                if learn.pbc:  # check for periodic boundary conditions
                    factor = s[learn.n-1]*s[0]
                else:
                    factor = 0
            else:
                s[j-1] = -1*s[j-1]  # reverse flip done by previous iteration
                factor = s[j]*s[j+1]  # calculate sigma z factor for the z term
            s[j] = -1*s[j]  # flip spin at current site
            self.calculateWave(s)  # find new wavefunction
            newWave = self.psi
            # calculate sigma x and add to x term
            xterm = xterm + (np.conj(newWave/oldWave))
            zterm = zterm + factor  # add factor to z term
        energy = -(learn.g*xterm)-zterm  # compute the energy
        return energy

    def calculateWave(self, s):
        '''
        s: Spin states in vector should be 1 and -1s

        a_vec: a_vec paramter

        b_vec: b_vec parameter

        w_mat: w_mat parameter 

        Returns: psi otan

        Creates the psi wavefunction as well as the otan values and returns them
        psi should be a value and otan should be a vector the size of learn.n
        '''
        self.psi = np.multiply(np.prod(
            2*np.cosh(np.add(self.b_vec, np.matmul(self.w_mat, s)))), np.exp(np.matmul(self.a_vec.T, s)))
        self.otan = np.tanh(np.add(self.b_vec.T, np.matmul(self.w_mat, s).T))

    def makeIntialValues(self):
        '''
        learn.n: Numer of particles 

        learn.m: Number of hidden nodes

        learn.exact: True if you want the learn.exact solution for the cluster state hamiltoniain False for np.random values

        Returns: self.a_vec self.b_vec self.w_mat

        This Function creates the 3 inital parameters for the descent.
        '''
        np.random.seed(53)
        if learn.exact:
            self.a_vec = np.zeros((learn.n, 1))
            self.b_vec = (1j*pi/4)*np.np.ones((learn.m, 1))
            self.w_mat = (
                1j*pi/4)*(spdiags([2*np.np.ones((learn.n)), 3*np.ones((learn.n)), np.ones((learn.n))], [-1, 0, 1], learn.n, learn.n).toarray())
            self.w_mat[learn.n-1, 0] = 1j*pi/4
            self.w_mat[0, learn.n-1] = 1j*2*pi/4
        else:
            self.a_vec = (np.random.rand(learn.n, 1)-0.5) + \
                (1j*(np.random.rand(learn.n, 1)-0.5))
            self.a_vec = self.a_vec/np.linalg.norm(self.a_vec, 2)

            self.b_vec = (np.random.rand(learn.m, 1)-0.5) + \
                (1j*(np.random.rand(learn.m, 1)-0.5))
            self.b_vec = self.b_vec/np.linalg.norm(self.b_vec, 2)

            self.w_mat = (1j*(np.random.rand(learn.n, learn.m)-0.5)) + \
                (1*(np.random.rand(learn.n, learn.m)-0.5))
            self.w_mat = self.w_mat/np.linalg.norm(self.w_mat, 2)

    def calculateGradientsLinear(self):
        '''
        a_vec: a_vec paramter

        b_vec: b_vec parameter

        w_mat: w_mat parameter

        metroLoops: Number of Metro loops

        burn: How many times it burns the iterations before calculating the values

        hamiltonian: Which hamiltonian the energy is being calculated for

        g: For tfim 

        walks: How many metrohasting walks is done

        pbc: Period Boundary Condition (True,False)

        Returns: gradA, gradB, gradW, E

        This function calculates the gradients for a,b,w and then returns those values as well as the 
        energy. This is done through metrohasting sampling.

        '''

        # Initialize all arrays for gradient calculation
        o_w = np.zeros((learn.n*learn.m, 1))
        o_w_e = np.zeros((learn.n*learn.m, 1))
        o_a = np.zeros(learn.n)
        o_a_e = np.zeros(learn.n)
        o_b = np.zeros(learn.m)
        o_b_e = np.zeros(learn.m)

        # Loop through desired number of walks
        for _ in range(learn.walks):
            sOld = np.random.rand(learn.n, 1)  # generate random v
            # Convert v and -1 and 1
            for j in range(learn.n):
                if sOld[j] < 0.5:
                    sOld[j] = -1
                else:
                    sOld[j] = 1
            # calculate starting wavefunction
            self.calculateWave(sOld)
            oldWave = self.psi
            oldOtan = self.otan
            r = np.random.randint(low=0, high=learn.n,
                                  size=(learn.metroLoops, 1))
            # Begin Metropolis Hastings
            for i in range(learn.metroLoops):

                # copy v #sNew should be copied onto gpu
                # PSA Python can be stupid and when something is set to another variable
                # it is done by pointer so if the old value changes it also changes the new
                # value. This can be fixed by doing a deepcopy
                sNew = copy.deepcopy(sOld)
                # generate random index to flip
                sNew[r[i]] = -sNew[r[i]]  # flip index
                # find new wavefunction
                tempSNew = sNew
                self.calculateWave(tempSNew)
                newWave = self.psi
                newOtan = self.otan
                # calculate ratio of probabilities
                ratio = abs(((newWave)/(oldWave)) ** 2)

                # Compare probabilities
                if ratio >= 1:
                    # New v accepted
                    oldWave = copy.deepcopy(newWave)
                    sOld = copy.deepcopy(sNew)
                    oldOtan = copy.deepcopy(newOtan)
                else:
                    ran = np.random.random()  # generate random double
                    if ratio > ran:
                        # New v accepted
                        oldWave = copy.deepcopy(newWave)
                        sOld = copy.deepcopy(sNew)
                        oldOtan = copy.deepcopy(newOtan)

                # Do not perform energy and gradient calculation if burning
                if i > learn.burn:

                    # Calculate energy of current v using hamiltonian
                    # PSA 2. Python also stupidly has all variables of a function pass by
                    # refrence. This means that the function can edit what is passed into it
                    # since this is not what we want a copy is made before hand
                    sTemp = copy.deepcopy(sOld)
                    epsilon = self.hamiltonian(sTemp, oldWave)
                    # Add current loops results to gradient calculation
                    votan = np.matmul(oldOtan.T, sOld.T)
                    revotan = np.reshape(votan, [learn.n*learn.m, 1])
                    o_w = o_w + revotan
                    o_w_e = o_w_e + np.conj(epsilon*(revotan))
                    o_a = o_a + sOld.T
                    o_a_e = o_a_e + (np.conj((epsilon)*(sOld))).T
                    o_b = o_b + oldOtan
                    o_b_e = o_b_e + np.conj(epsilon*(oldOtan))

                    # Add current energy to total energy
                    self.E = self.E+epsilon

        # calculate total number of loops used
        loops = learn.walks*(learn.metroLoops-1-learn.burn)

        # Average everything by the number of loops
        self.E = self.E/loops
        o_w = o_w/loops
        o_w_e = o_w_e/loops
        o_a = o_a/loops
        o_a_e = o_a_e/loops
        o_b = o_b/loops
        o_b_e = o_b_e/loops

        # Calculate gradients
        self.gradA = ((o_a_e) - np.conj(o_a)*(self.E)).T
        self.gradB = ((o_b_e) - np.conj(o_b)*(self.E)).T
        self.gradW = ((o_w_e) - np.conj(o_w)*(self.E))
        self.gradW = np.reshape(self.gradW.T, [learn.n, learn.m])

        # normA = np.linalg.norm( self.gradA,2)
        # normB = np.linalg.norm( self.gradB,2)
        # normW = np.linalg.norm( self.gradW,2)

        # self.gradA=self.gradA/normA
        # self.gradB=self.gradB/normB
        # self.gradW=self.gradW/normW

        # take only the real part of the energy to pass out
        self.E = np.real(self.E)

    def calculateGradientsSGD(self):
        '''
        a_vec: a_vec paramter

        b_vec: b_vec parameter

        w_mat: w_mat parameter

        metroLoops: Number of Metro loops

        burn: How many times it burns the iterations before calculating the values

        hamiltonian: Which hamiltonian the energy is being calculated for

        g: For tfim 

        walks: How many metrohasting walks is done

        pbc: Period Boundary Condition (True,False)

        Returns: gradA, gradB, gradW, E

        This function calculates the gradients for a,b,w and then returns those values as well as the 
        energy. This is done through metrohasting sampling.

        '''

        # Initialize all arrays for gradient calculation
        o_w = np.zeros((learn.n*learn.m, 1))
        o_w_e = np.zeros((learn.n*learn.m, 1))
        o_a = np.zeros(learn.n)
        o_a_e = np.zeros(learn.n)
        o_b = np.zeros(learn.m)
        o_b_e = np.zeros(learn.m)

        S_a = np.zeros((learn.n, learn.n))
        S_b = np.zeros((learn.m, learn.m))
        S_w = np.zeros((learn.n*learn.m, learn.n*learn.m))

        # Loop through desired number of walks
        for _ in range(learn.walks):
            sOld = np.random.rand(learn.n, 1)  # generate random v
            # Convert v and -1 and 1
            for j in range(learn.n):
                if sOld[j] < 0.5:
                    sOld[j] = -1
                else:
                    sOld[j] = 1
            # calculate starting wavefunction
            self.calculateWave(sOld)
            oldWave = self.psi
            oldOtan = self.otan
            r = np.random.randint(low=0, high=learn.n,
                                  size=(learn.metroLoops, 1))
            # Begin Metropolis Hastings
            for i in range(learn.metroLoops):

                # copy v #sNew should be copied onto gpu
                # PSA Python can be stupid and when something is set to another variable
                # it is done by pointer so if the old value changes it also changes the new
                # value. This can be fixed by doing a deepcopy
                sNew = copy.deepcopy(sOld)
                # generate random index to flip
                sNew[r[i]] = -sNew[r[i]]  # flip index
                # find new wavefunction
                tempSNew = sNew
                self.calculateWave(tempSNew)
                newWave = self.psi
                newOtan = self.otan
                # calculate ratio of probabilities
                ratio = abs(((newWave)/(oldWave)) ** 2)

                # Compare probabilities
                if ratio >= 1:
                    # New v accepted
                    oldWave = copy.deepcopy(newWave)
                    sOld = copy.deepcopy(sNew)
                    oldOtan = copy.deepcopy(newOtan)
                else:
                    ran = np.random.random()  # generate random double
                    if ratio > ran:
                        # New v accepted
                        oldWave = copy.deepcopy(newWave)
                        sOld = copy.deepcopy(sNew)
                        oldOtan = copy.deepcopy(newOtan)

                # Do not perform energy and gradient calculation if burning
                if i > learn.burn:

                    # Calculate energy of current v using hamiltonian
                    # PSA 2. Python also stupidly has all variables of a function pass by
                    # refrence. This means that the function can edit what is passed into it
                    # since this is not what we want a copy is made before hand
                    sTemp = copy.deepcopy(sOld)
                    epsilon = self.hamiltonian(sTemp, oldWave)
                    # Add current loops results to gradient calculation
                    votan = np.matmul(oldOtan.T, sOld.T)
                    revotan = np.reshape(votan, [learn.n*learn.m, 1])
                    o_w = o_w + revotan
                    o_w_e = o_w_e + np.conj(epsilon*(revotan))
                    o_a = o_a + sOld.T
                    o_a_e = o_a_e + (np.conj((epsilon)*(sOld))).T
                    o_b = o_b + oldOtan
                    o_b_e = o_b_e + np.conj(epsilon*(oldOtan))

                    S_a = S_a + np.matmul(np.conj(sOld), sOld.T)
                    S_b = S_b + np.matmul(np.conj(oldOtan).T, oldOtan)
                    S_w = S_w + np.matmul(np.conj(revotan), revotan.T)

                    # Add current energy to total energy
                    self.E = self.E+epsilon

        # calculate total number of loops used
        loops = learn.walks*(learn.metroLoops-1-learn.burn)

        # Average everything by the number of loops
        self.E = self.E/loops
        o_w = o_w/loops
        o_w_e = o_w_e/loops
        o_a = o_a/loops
        o_a_e = o_a_e/loops
        o_b = o_b/loops
        o_b_e = o_b_e/loops

        # Finish calculation stochastic matrices
        S_a = S_a - np.matmul(np.conj(o_a), o_a.T)
        S_b = S_b - np.matmul(np.conj(o_b), o_b.T)
        S_w = S_w - np.matmul(np.conj(o_w), o_w.T)
        S_a = S_a/loops
        S_b = S_b/loops
        S_w = S_w/loops

        # To help and try and prevent pinv from breaking later
        for i in range(learn.n):
            S_a[i, i] = S_a[i, i]+.001*S_a[i, i]
            S_b[i, i] = S_b[i, i]+.001*S_b[i, i]
        for i in range(learn.n*learn.m):
            S_w[i, i] = S_w[i, i]+.001*S_w[i, i]

        # Calculate gradients
        self.gradA = ((o_a_e) - np.conj(o_a)*(self.E)).T
        self.gradB = ((o_b_e) - np.conj(o_b)*(self.E)).T
        self.gradW = ((o_w_e) - np.conj(o_w)*(self.E))
        self.gradA = np.matmul(np.linalg.pinv(S_a), self.gradA)
        self.gradB = np.matmul(np.linalg.pinv(S_b), self.gradB)
        self.gradW = np.matmul(np.linalg.pinv(S_w), self.gradW)
        self.gradW = np.reshape(self.gradW.T, [learn.n, learn.m])

        # normA = np.linalg.norm( self.gradA,2)
        # normB = np.linalg.norm( self.gradB,2)
        # normW = np.linalg.norm( self.gradW,2)

        # self.gradA=self.gradA/normA
        # self.gradB=self.gradB/normB
        # self.gradW=self.gradW/normW

        # take only the real part of the energy to pass out
        self.E = np.real(self.E)
