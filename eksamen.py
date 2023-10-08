import numpy as np
import matplotlib.pyplot as plt
from math import atan2
from mpl_toolkits import mplot3d
from scipy.optimize import curve_fit
from numba import jit


# Constants, these are modified throughout the tasks
J = 1  # meV
dz = 0.1  # meV
ys = 5.788*10**-2  # J/T
ysB0 = 0.1  # J
B0 = ysB0/ys  # T
alpha = 0.01  # Damping constant
gyromagnetic = 0.176  # 1/Tps
dt = 1*10**-3  # ps
kb = 1.38*10**-23
#kbT = 0.05


# Question a)

@jit(nopython=True, cache=True)
def normalize(spins):
    """Normalizes an array consisting of three spin components
    by projecting them onto the unit sphere."""
    pLength = np.sqrt(spins[0]**2 + spins[1]**2 + spins[2]**2)
    spins /= pLength
    return spins


# N=1, T=0, J=0, a=0, either dz or B
@jit(nopython=True, cache=True)
def LLG(spins):
    """Solves the Landau-Lifschitz-Gilbert equation for a single particle."""
    Fj = np.array([0, 0, 2*dz*spins[2]])/ys
    dS = -gyromagnetic/(1+alpha**2) * (np.cross(spins, Fj) + np.cross(alpha*spins, np.cross(spins, Fj)))
    return dS


@jit(nopython=True, cache=True)
def singleSpinHeun(spins):
    """Implements the Heun method to estimate the change in spin."""
    predictedSpins = spins + dt * LLG(spins)
    newSpins = spins + dt/2 * (LLG(spins) + LLG(predictedSpins))
    normalizedSpins = normalize(newSpins)
    return normalizedSpins


@jit(nopython=True, cache=True)
def evolveSingleSpin(spin, steps):
    """This function is used to plot the spin data for the single particle."""
    spin = normalize(spin)
    timeArray = np.zeros(steps)
    spinArrayX = np.ones(steps)
    spinArrayY = np.ones(steps)
    spinArrayZ = np.ones(steps)
    spinArrayX[0], spinArrayY[0], spinArrayZ[0] = spin[0], spin[1], spin[2]
    for i in range(1, steps):
        spin = singleSpinHeun(spin)
        timeArray[i] = i*dt
        spinArrayX[i] = spin[0]
        spinArrayY[i] = spin[1]
        spinArrayZ[i] = spin[2]
    return timeArray, spinArrayX, spinArrayY, spinArrayZ


def plotSingleSpin(time, xSpins, ySpins, zSpins):
    """Plots the components of the spin over time."""
    plt.plot(time, xSpins, label='x component')
    plt.plot(time, ySpins, label='y-component')
    plt.plot(time, zSpins, label='z-component')
    #plt.ylim((-0.105, 0.13))
    plt.xlabel('Time [ps]', fontsize=12)
    plt.ylabel('Normalized spin component', fontsize=12)
    plt.legend()
    #plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()


def plotSingleSpin3D(xSpins, ySpins, zSpins):
    """Plots the spin of a single particle onto the unit sphere."""
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(100), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.2)
    ax.scatter(xSpins, ySpins, zSpins, marker='.', s=3)
    ax.grid(False)
    ax.set_axis_off()
    ax.set_box_aspect(aspect=(1, 1, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)


def curveFunc(t, a, omega, tau):
    """Function used with the scipy.omptimize.curve_fit function."""
    return a*np.cos(omega*t)*np.exp(-t/tau)


@jit(nopython=True, cache=True)
def chainLLG(this, neighbours):
    """Solves the Landau-Lifschitz-Gilbert equation for a chain of spins.
    This is done by including nearest neighbour interactions."""
    Fj = 1/ys * (J * (neighbours[0] + neighbours[1]) + np.array([0, 0, 2*dz*this[2]]))
    dS = -gyromagnetic/(1+alpha**2) * (np.cross(this, Fj) + np.cross(alpha*this, np.cross(this, Fj)))
    return dS


@jit(nopython=True, cache=True)
def makePredictions(spinChain, isPeriodic):
    """Returns an array consisting of the predicted spins of all the particles in the chain.
    The predictions are made by using the first step of the Heun method."""
    predictions = np.zeros((len(spinChain), 3))
    for i in range(len(spinChain)):
        neighbours = checkIndex1D(spinChain, i, isPeriodic)
        spinPrediction = spinChain[i] + dt * chainLLG(spinChain[i], neighbours)
        predictions[i] = spinPrediction
    return predictions


@jit(nopython=True, cache=True)
def checkIndex1D(chain, i, isPeriodic):
    """Ensures that the indexing does noe go out of range. Can account for both
    periodic and non-periodic conditions."""
    neighbours = np.zeros((2, 3))
    if isPeriodic:
        if i == 0:
            left = chain[-1]
            right = chain[i+1]
        elif i == len(chain)-1:
            left = chain[i-1]
            right = chain[0]
        else:
            left = chain[i-1]
            right = chain[i+1]
        neighbours[0] = left
        neighbours[1] = right
    else:
        if i == 0:
            left = np.zeros(3)
            right = chain[i+1]
        elif i == len(chain)-1:
            left = chain[i-1]
            right = np.zeros(3)
        else:
            left = chain[i-1]
            right = chain[i+1]
        neighbours[0] = left
        neighbours[1] = right
    return neighbours


@jit(nopython=True, cache=True)
def chainHeun(spinArray, preds, isPeriodic):
    """Implements the second step of the heun method for a chain of particles."""
    newSpins = np.ones((len(spinArray), 3))
    for i in range(len(spinArray)):
        neighbours = checkIndex1D(spinArray, i, isPeriodic)
        predNeighbours = checkIndex1D(preds, i, isPeriodic)
        newSpin = spinArray[i] + dt/2 * (chainLLG(spinArray[i], neighbours) +
                                         chainLLG(preds[i], predNeighbours))
        newSpin = normalize(newSpin)
        newSpins[i] = newSpin
    return newSpins


@jit(nopython=True, cache=True)
def createInitialSpinChain(N):
    """Creates the initial spin conditions in the chain of particles.
    All particles are set to have spin in only the z-direction. One particle is
    then excited by shifting the spin."""
    spinChain = np.ones((N, 3))
    for i in range(N):
        spinChain[i] = np.array([0, 0, 1])
    excitation = normalize(np.array([0.1, 0.2, 0.5]))
    spinChain[0] = excitation
    return spinChain


@jit(nopython=True, cache=True)
def createRandomSpinChain(N):
    """Creates a spin chain where all particles gets assigned a random spin direction."""
    spinChain = np.ones((N, 3))
    for i in range(N):
        x, y, z = np.random.uniform(-1, 1, 3)
        spin = normalize(np.array([x, y, z]))
        spinChain[i] = spin
        while x**2 + y**2 + z**2 > 1:
            x, y, z = np.random.uniform(-1, 1, 3)
            spin = normalize(np.array([x, y, z]))
            spinChain[i] = spin
    return spinChain


@jit(nopython=True, cache=True)
def evolveSpinChain(spinChain, steps, isPeriodic):
    """Updates a initial spin chain for each time step."""
    matrix = np.ones((steps, len(spinChain), 3))
    matrix[0] = spinChain
    for i in range(steps-1):
        predictedSpins = makePredictions(spinChain, isPeriodic)
        spinChain = chainHeun(spinChain, predictedSpins, isPeriodic)
        matrix[i+1] = spinChain
    return matrix


def plotChain(particles, steps, isPeriodic):
    """Used for plotting the spin chain."""
    time = np.linspace(0, steps*dt, steps)
    spinChain = createInitialSpinChain(particles)
    mat = evolveSpinChain(spinChain, steps, isPeriodic)
    xMat = np.ones((steps, particles))
    yMat = np.ones((steps, particles))
    zMat = np.ones((steps, particles))
    for i in range(steps):
        for j in range(particles):
            xMat[i][j] = mat[i][j][0]
            yMat[i][j] = mat[i][j][1]
            zMat[i][j] = mat[i][j][2]
    #plt.imshow(zMat, origin='lower', aspect='auto')
    #plt.colorbar()
    #plt.clim(-1, 1)
    #plt.xlabel('z component of spins', fontsize=12)
    #plt.ylabel('Time steps', fontsize=12)
    #fig, ax = plt.subplots(1, 3, sharey=True)
    #ax[0].set(xlabel='x component', ylabel='Time steps')
    #ax[1].set(xlabel='y component')
    #ax[2].set(xlabel='z component')
    #ax[0].imshow(xMat, origin='lower', aspect='auto')
    #ax[1].imshow(yMat, origin='lower', aspect='auto')
    #ax[2].imshow(zMat, origin='lower', aspect='auto')
    #fig.tight_layout()
    #fig.savefig('Antiferro.png', dpi=300)


@jit(nopython=True, cache=True)
def latticeLLG(this, neighbors, kbT):
    """Implements the Landau-Lifschitz-Gilbert equation with the magnetic field
    and the thermal noise term."""
    Ferr = 1/ys * (J * (neighbors[0] + neighbors[1] + neighbors[2] + neighbors[3]) +
                   np.array([0, 0, 2*dz*this[2]]) + ys*np.array([0, 0, B0]))
    kbFactor = np.sqrt(2*alpha*kbT/(gyromagnetic*ys*dt))
    x = np.random.normal(0, 1)
    y = np.random.normal(0, 1)
    z = np.random.normal(0, 1)
    gauss = np.array([x, y, z])
    Fth = gauss * kbFactor
    Fj = Ferr + Fth
    dS = -gyromagnetic/(1+alpha**2) * (np.cross(this, Fj) + np.cross(alpha*this, np.cross(this, Fj)))
    return dS


@jit(nopython=True, cache=True)
def checkIndex2D(lattice, i, j):
    """Ensures that the indexing does not go outside of range.
    The function assumes periodic boundary conditions."""
    neighbors = np.ones((4, 3))
    if j == 0:
        left = lattice[i][-1]
    else:
        left = lattice[i][j-1]
    neighbors[0] = left
    if j == len(lattice)-1:
        right = lattice[i][0]
    else:
        right = lattice[i][j+1]
    neighbors[1] = right
    if i == 0:
        bottom = lattice[-1][j]
    else:
        bottom = lattice[i-1][j]
    neighbors[2] = bottom
    if i == len(lattice)-1:
        top = lattice[0][j]
    else:
        top = lattice[i+1][j]
    neighbors[3] = top
    return neighbors


@jit(nopython=True, cache=True)
def makePrediction2D(lattice, kbT):
    """Makes predictions for the spins of the particles in the lattice
    by using the fist step of the Heun method."""
    predictions = np.ones((len(lattice), len(lattice), 3))
    for i in range(len(lattice)):
        for j in range(len(lattice[0])):
            neighbors = checkIndex2D(lattice, i, j)
            pred = lattice[i][j] + dt * (latticeLLG(lattice[i][j], neighbors, kbT))
            predictions[i][j] = normalize(pred)
    return predictions


@jit(nopython=True, cache=True)
def heun2D(lattice, predictions, kbT):
    """Implements the second step of the Heun method."""
    newLattice = np.ones((len(lattice), len(lattice), 3))
    for i in range(len(lattice)):
        for j in range(len(lattice[0])):
            neighbors = checkIndex2D(lattice, i, j)
            predNeighbors = checkIndex2D(predictions, i, j)
            newSpin = lattice[i][j] + dt/2 * (latticeLLG(lattice[i][j], neighbors, kbT) +
                                              latticeLLG(predictions[i][j], predNeighbors, kbT))
            newLattice[i][j] = normalize(newSpin)
    return newLattice


@jit(nopython=True, cache=True)
def createRandomLattice(N):
    """Generates a random lattice. The throught behind the randomization is explained in the report."""
    lattice = np.ones((N, N, 3))
    for i in range(N):
        for j in range(N):
            x, y, z = np.random.uniform(-1, 1, 3)
            spin = normalize(np.array([x, y, z]))
            lattice[i][j] = spin
            while x**2 + y**2 + z**2 > 1:
                x, y, z = np.random.uniform(-1, 1, 3)
                spin = normalize(np.array([x, y, z]))
                lattice[i][j] = spin
    return lattice


@jit(nopython=True, cache=True)
def createUpLattice(N):
    """Creates a lattice consisting of spins pointing in the positive z-direction."""
    lattice = np.ones((N, N, 3))
    for i in range(N):
        for j in range(N):
            spin = np.array([0, 0, 1])
            lattice[i][j] = spin
    return lattice


@jit(nopython=True, cache=True)
def twoDimensional(lattice, cycles, kbT):
    """Updates the spins of the particles in the lattice for each time step."""
    time = np.zeros(cycles)
    magnetization = np.zeros(cycles)
    magnetization[0] = len(lattice)**2
    pred = makePrediction2D(lattice, kbT)
    for t in range(cycles-1):
        lattice = heun2D(lattice, pred, kbT)
        pred = makePrediction2D(lattice, kbT)
        for i in range(len(lattice)):
            for j in range(len(lattice)):
                magnetization[t+1] += lattice[i][j][2]
        time[t+1] = dt*(t+1)
    magnetization /= len(lattice)**2
    return lattice, time, magnetization


def getMagnetization(particles, cycles, maxTemp, tempInterval):
    """Finds the temperature-dependent magnetization of the lattice for
    various temperatures."""
    avgTemperatures = np.ones(tempInterval)
    stdTemperatures = np.ones(tempInterval)
    kbTs = np.linspace(0, maxTemp, tempInterval)
    for i in range(tempInterval):
        mat, t, mag = twoDimensional(createUpLattice(particles), cycles, kbTs[i])
        avgTemperatures[i] = np.average(mag[5000:])
        stdTemperatures[i] = np.std(mag[5000:])
        print(i, ': ', avgTemperatures[i])
    np.savetxt('averageTemperatures25oneoverfiveTimes.txt', avgTemperatures)
    np.savetxt('stdTemperatures25oneoverfiveTimes.txt', stdTemperatures)


def plotMagnetization(maxTemp, iterations):
    """Plots the magnetization over a given interval of temperatures."""
    temp = maxTemp*1.6*10**-22/kb
    kbTs = np.linspace(0, temp, iterations)
    plt.xlim((-2, temp))
    averageTs = np.loadtxt('averageTemperatures25oneoverfiveTimes.txt')
    stdTs = np.loadtxt('stdTemperatures25oneoverfiveTimes.txt')
    upperLim = averageTs + stdTs
    lowerLim = averageTs - stdTs
    plt.scatter(kbTs, averageTs, color='darkblue', marker='.', s=8, label='Measurements')
    plt.plot(kbTs, upperLim, color='steelblue')
    plt.plot(kbTs, lowerLim, color='steelblue')
    plt.fill_between(kbTs, upperLim, lowerLim, color='steelblue', alpha=0.5, label='Standard deviation')
    plt.plot(kbTs, np.zeros(len(kbTs)), '--', color='grey')
    plt.xlabel('Temperature [K]', fontsize=12)
    plt.ylabel('Magnetization', fontsize=12)
    plt.plot(np.ones(100)*22, np.linspace(0, 1, 100), '--', label='$T_c$ = 22 K')
    plt.legend()
    plt.tight_layout()
    #plt.errorbar(kbTs, averageTs, stdTs, fmt='.')


plt.show()
