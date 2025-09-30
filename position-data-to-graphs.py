import os
import numpy as np
from scipy.optimize import curve_fit as curveFit
from sklearn.metrics import r2_score as r2Score
import matplotlib.pyplot as plt
from math import log10, floor


# Gets the folder where all the position data is stored
dataFolderPath = os.path.abspath(__name__).removesuffix("__main__")+"position-data\\"

# Gets the folder where all the position graphs are is stored
positionFolderPath = os.path.abspath(__name__).removesuffix("__main__")+"position-graphs\\"

# Gets the folder where all the period graphs are is stored
periodFolderPath = os.path.abspath(__name__).removesuffix("__main__")+"period-graphs\\"

# Gets the folder where all the q-factor graphs are stored
qFactorFolderPath = os.path.abspath(__name__).removesuffix("__main__")+"q-factor-graphs\\"

# Gets the folder where all the peaks graphs are stored
peaksFolderPath = os.path.abspath(__name__).removesuffix("__main__")+"peaks-graphs\\"

# Gets the frame rate
frameRate = 60
with open(os.path.abspath(__name__).removesuffix("__main__")+"frame-rate.txt") as file:
    lines = file.readlines()
    frameRate = int(lines[2])


# Initial angle and length uncertainties
initialAngleUncertainty = 0
lengthUncertainty = 0
with open(os.path.abspath(__name__).removesuffix("__main__")+"uncertainties.txt") as file:
    lines = file.readlines()
    lengthUncertainty = float(lines[0].split()[1])
    initialAngleUncertainty = float(lines[1].split()[1])


# Stores the data in format: [(length 1, initial angle 1, period 1, dampening 1, q-factor 1, period fit 1, amplitude fit 1),
#                             (length 2, initial angle 2, period 2, dampening 2, q-factor 2, period fit 2, amplitude fit 2), ...]
data = []


# Defines rounding functions for uncertainties:
def roundUncertainties(value, uncertainty):
    uncertainty = round(uncertainty, -int(floor(log10(abs(uncertainty)))))

    if uncertainty >= 1:
        uncertainty = int(uncertainty+0.5)

    value = round(value, -int(floor(log10(uncertainty))))

    if value >= 1 and uncertainty >= 1:
        value = int(value+0.5)

    return str(value) + " ±" + str(uncertainty)

# Defines the damped harmonic motion function
def dampedHarmonicMotion(time, initialAngle, dampening, period, phaseShift, bias):
    return initialAngle*np.e**(-time/dampening)*np.cos(2*np.pi*time/period+phaseShift)+bias

def exponentialDecay(time, initialAngle, dampening):
    return initialAngle*np.e**(-time/dampening)

# Iterates through all data
for fileName in os.listdir(dataFolderPath):
    filePath = dataFolderPath+fileName

    # Variables for all properties
    length = None
    initialAngle = None
    periodFit = None
    dampeningFactor = None
    qFactor = None
    bias = None
    period = None
    qUnc = None

    # Reads length and initial angle from file name
    temp = fileName.split("_")
    length = float(temp[0] + "." + temp[1])
    initialAngle = float(temp[2] + "." + temp[3])*np.pi/180

    # Reads the data from the file
    positionTimeData = []
    with open(filePath) as file:
        for line in file.readlines():
            temp = line.split(" ")
            positionTimeData.append([float(temp[0]), float(temp[1])])

    # Fits the data to the damped harmonic motion model
    xData = [i[0] for i in positionTimeData]
    yData = [i[1] for i in positionTimeData]

    params, covariance = curveFit(dampedHarmonicMotion, xData, yData, maxfev=10000)
    angleFit, dampeningFactor, periodFit, phaseShift, bias = params

    qFactor = abs(np.pi*dampeningFactor/periodFit)

    # Gets uncertainties for dampening factor and q factor
    dampeningUnc = np.sqrt(np.diagonal(covariance))[1]
    periodFitUnc = np.sqrt(np.diagonal(covariance))[2]
    qUnc = np.pi*dampeningFactor/periodFit*max(dampeningUnc/dampeningFactor, periodFitUnc/periodFit)

    # Removes bias
    yData -= bias

    # Gets the period
    startTime = -11
    endTime = -11
    lastAngle = yData[0]
    for t, a in list(zip(xData, yData)):
        if t-startTime < 0.3:
            lastAngle = a
            continue
        if t-endTime < 0.3:
            lastAngle = a
            continue
        if np.sign(lastAngle) != np.sign(a):
            if startTime == -11:
                startTime = t
            elif endTime == -11:
                endTime = t
            else:
                endTime = t
                break
        lastAngle = a

    period = (endTime-startTime)

    # Gets the correct q factor
    xPeaks = []
    yPeaks = []

    for i in range(1, len(xData)-1):
        if max(yData[i], yData[i-1], yData[i+1]) == yData[i]:
            xPeaks.append(xData[i])
            yPeaks.append(yData[i])
    
    params, covariance = curveFit(exponentialDecay, xPeaks, yPeaks, maxfev=10000)
    dampeningFactor = params[1]
    dampeningUnc = np.sqrt(np.diagonal(covariance))[1]
    qFactor = abs(np.pi*dampeningFactor/period)
    qUnc = np.pi*dampeningFactor/period*max(dampeningUnc/dampeningFactor, 1/(2*frameRate)/period)


    # Gets q factor using method 2
    firstPeak = False
    firstPeakAmp = 0
    finalPeakAmp = 0
    cycles = 0
    for i in range(1, len(xData)-1):
        if max(yData[i], yData[i-1], yData[i+1]) == yData[i]:
            if not firstPeak:
                firstPeakAmp = yData[i]
                finalPeakAmp = yData[i]
                firstPeak = True
            else:
                cycles += 1
                finalPeakAmp = yData[i]
    
    qFactor2 = (-cycles*np.pi)/np.log(finalPeakAmp/firstPeakAmp)
    qUnc2 = 0


    # Adds the data to the main dataset
    data.append([length, initialAngle, period, dampeningFactor, qFactor, periodFit, qUnc, qFactor2, qUnc2])


# Converts data to format: (length, initial angle): [[period 1, period 2, ...], [dampening factor 1, dampening factor 2, ...], [q factor 1, q factor 2, ...], [period fit 1, period fit 2, ...]]
dataFrame = {}
for row in data:
    index = (row[0], row[1])
    if index in dataFrame:
        dataFrame[index][0].append(row[2])
        dataFrame[index][1].append(row[3])
        dataFrame[index][2].append(row[4])
        dataFrame[index][3].append(row[7])
        dataFrame[index][4].append(row[5])
        dataFrame[index][5].append(row[6])
        dataFrame[index][6].append(row[8])
    else:
        dataFrame[index] = [[row[2]], [row[3]], [row[4]], [row[7]], [row[5]], [row[6]], [row[8]]]

# Gets the means in format: (length, initial angle): [period mean, dampening factor mean, q factor mean]
means = {}
for index in dataFrame:
    means[index] = [sum(dataFrame[index][i])/len(dataFrame[index][i]) for i in range(4)]


# Gets the standard deviations in format: (length, initial angle): [period std dev, dampening factor std dev, q factor std dev]
def getStandardDeviation(mean, sample):
    if len(sample) <= 1:
        return 0
    return (sum([(i-mean)**2 for i in sample])/(len(sample)-1))**0.5

stdDevs = {}
for index in dataFrame:
    stdDevs[index] = [getStandardDeviation(means[index][i], dataFrame[index][i]) for i in range(4)]


# Gets the uncertainty of the mean in format: (length, initial angle): [period uncertainty, dampening factor uncertainty, q factor uncertainty]
uncertaintyMean = {}
for index in dataFrame:
    uncertaintyMean[index] = [stdDevs[index][i]/(len(dataFrame[index][i])**0.5) for i in range(4)]


# Gets the type B uncertainties in format: (length, initial angle): [period uncertainty, dampening factor uncertainty, q factor uncertainty]
uncertaintyB = {}
for index in dataFrame:
    uncertaintyB[index] = [1/(2*frameRate), 0, 0, 0]


# Saves all position-time graphs
for fileName in os.listdir(dataFolderPath):
    filePath = dataFolderPath+fileName

    # Variables for all properties
    length = None
    initialAngle = None
    angleFit = None
    periodFit = None
    dampeningFactor = None
    qFactor = None
    bias = None
    qUnc = None

    # Reads length and initial angle from file name
    temp = fileName.split("_")
    length = float(temp[0] + "." + temp[1])
    initialAngle = float(temp[2] + "." + temp[3])*np.pi/180

    # Reads the data from the file
    positionTimeData = []
    with open(filePath) as file:
        for line in file.readlines():
            temp = line.split(" ")
            positionTimeData.append([float(temp[0]), float(temp[1]), float(temp[2])])

    # Fits the data to the damped harmonic motion model
    xData = [i[0] for i in positionTimeData]
    yData = [i[1] for i in positionTimeData]
    yErrorData = [i[2] for i in positionTimeData]

    params, covariance = curveFit(dampedHarmonicMotion, xData, yData, maxfev=1000000)
    angleFit, dampeningFactor, periodFit, phaseShift, bias = params
    periodFit = abs(periodFit)

    qFactor = abs(np.pi*dampeningFactor/periodFit)

    # Gets uncertainties for dampening factor and q factor
    angleFitUnc = np.sqrt(np.diagonal(covariance))[0]
    dampeningUnc = np.sqrt(np.diagonal(covariance))[1]
    periodFitUnc = np.sqrt(np.diagonal(covariance))[2]
    phaseShiftUnc = np.sqrt(np.diagonal(covariance))[3]

    # Removes bias
    yData -= bias

    # Gets the period
    startTime = -11
    endTime = -11
    lastAngle = yData[0]
    for t, a in list(zip(xData, yData)):
        if t-startTime < 0.3:
            lastAngle = a
            continue
        if t-endTime < 0.3:
            lastAngle = a
            continue
        if np.sign(lastAngle) != np.sign(a):
            if startTime == -11:
                startTime = t
            elif endTime == -11:
                endTime = t
            else:
                endTime = t
                break
        lastAngle = a

    period = (endTime-startTime)

    # Gets the correct q factor
    xPeaks = []
    yPeaks = []
    peaksErrorData = []

    for i in range(1, len(xData)-1):
        if max(yData[i], yData[i-1], yData[i+1]) == yData[i]:
            xPeaks.append(xData[i])
            yPeaks.append(yData[i])
            peaksErrorData.append(yErrorData[i])
    
    params, covariance = curveFit(exponentialDecay, xPeaks, yPeaks, maxfev=10000)
    dampeningFactor = params[1]
    dampeningUnc = np.sqrt(np.diagonal(covariance))[1]
    qFactor = abs(np.pi*dampeningFactor/period)
    qUnc = np.pi*dampeningFactor/period*max(dampeningUnc/dampeningFactor, 1/(2*frameRate)/period)

    uncertaintyB[(length, initialAngle)][2] = max(uncertaintyB[(length, initialAngle)][2], qUnc)

    # Gets q factor using method 2
    firstPeak = False
    firstPeakAmp = 0
    firstPeakUnc = 0
    finalPeakAmp = 0
    finalPeakUnc = 0
    cycles = 0
    for i in range(1, len(xData)-1):
        if max(yData[i], yData[i-1], yData[i+1]) == yData[i]:
            if not firstPeak:
                firstPeakAmp = yData[i]
                finalPeakAmp = yData[i]
                firstPeakUnc = yErrorData[i]
                finalPeakUnc = yErrorData[i]
                firstPeak = True
            else:
                cycles += 1
                finalPeakAmp = yData[i]
                finalPeakUnc = yErrorData[i]
    
    qFactor2 = (-cycles*np.pi)/np.log(finalPeakAmp/firstPeakAmp)
    qUnc2 = qFactor2 * min(firstPeakUnc/firstPeakAmp, finalPeakUnc/finalPeakAmp)
    qUnc2 = abs(qUnc2)

    uncertaintyB[(length, initialAngle)][3] = max(uncertaintyB[(length, initialAngle)][3], qUnc2)

    print("Graph Name: "+fileName[:-9]+"-graph.png")
    print("Control Variable (Initial Angle [rad]):", roundUncertainties(initialAngle, initialAngleUncertainty))
    print("Control Variable (Length [cm]):", roundUncertainties(length, initialAngleUncertainty))
    print("Q Factor Using Method 1:", roundUncertainties(qFactor, qUnc))
    print("Q Factor Using Method 2:", roundUncertainties(qFactor2, qUnc2))
    print("\n\n")

    plt.errorbar(xData, yData, xerr = np.full(len(xData), 1/(2*frameRate)), yerr = yErrorData, fmt='o', ms=3)
    plt.xlabel("Time [s]", fontsize = 12)
    plt.ylabel("Angle [rad]", fontsize = 12)
    
    fig = plt.gcf()
    fig.set_size_inches(25, 5)
    fig.savefig(positionFolderPath+fileName[:-9]+"-graph.png", dpi=200)
    plt.clf()

    plt.errorbar(xPeaks, yPeaks, xerr = np.full(len(xPeaks), 1/(2*frameRate)), yerr = peaksErrorData, fmt='o', ms=3)
    plt.xlabel("Time [s]", fontsize = 12)
    plt.ylabel("Angle [rad]", fontsize = 12)
    
    plt.plot(np.arange(min(xPeaks), max(xPeaks), 0.001), [exponentialDecay(x, params[0], params[1]) for x in np.arange(min(xPeaks), max(xPeaks), 0.001)])
    fig = plt.gcf()
    fig.set_size_inches(25, 5)
    fig.savefig(peaksFolderPath+fileName[:-9]+"-graph.png", dpi=200)
    plt.clf()


# Gets the total uncertainties in format: (length, initial angle): [period uncertainty, dampening factor uncertainty, q factor uncertainty]
uncertaintyTotal = {}
for index in dataFrame:
    uncertaintyTotal[index] = [max(uncertaintyMean[index][i], uncertaintyB[index][i]) for i in range(4)]


# Saves period-length graph
def powerLaw(length, k, n):
    return k*length**n

periodLengthData = {} # initialAngle: [xValues, yValues, x uncertainties, y uncertainties]
for index in means:
    if index[1] in periodLengthData:
        periodLengthData[index[1]][0].append(index[0])
        periodLengthData[index[1]][1].append(means[index][0])
        periodLengthData[index[1]][2].append(lengthUncertainty)
        periodLengthData[index[1]][3].append(uncertaintyTotal[index][0])
    else:
        periodLengthData[index[1]] = [[index[0]], [means[index][0]], [lengthUncertainty], [uncertaintyTotal[index][0]]]

for angle in periodLengthData:
    xValues, yValues, xUncertainties, yUncertainties = periodLengthData[angle]

    if len(xValues) <= 3:
        continue

    params, covariance = curveFit(powerLaw, xValues, yValues)
    k, n = params
    kUnc, nUnc = np.sqrt(np.diagonal(covariance))

    print("Graph Name: "+"-".join(str(float(angle)).split("."))+"-graph.png")
    print("Line of best fit parameters for T = K * Length ^ N")
    print("K Value:", roundUncertainties(k, kUnc))
    print("N Value:", roundUncertainties(n, nUnc))
    print("R^2 Value:", r2Score(yValues, np.array([powerLaw(x, k, n) for x in xValues])))
    print("Control Variable (Initial Angle [rad]):", roundUncertainties(angle, initialAngleUncertainty))
    print("\n\n")
    
    plt.errorbar(xValues, yValues, xerr = xUncertainties, yerr = yUncertainties, fmt='o')
    plt.xlabel("Length [cm]", fontsize = 12)
    plt.ylabel("Period [s]", fontsize = 12)

    plt.plot(np.arange(min(xValues), max(xValues), 0.001), [powerLaw(x, k, n) for x in np.arange(min(xValues), max(xValues), 0.001)])
    fig = plt.gcf()
    fig.set_size_inches(10, 7)
    fig.savefig(periodFolderPath+"period-length-graphs\\period-length-"+"-".join(str(float(angle)).split("."))+"-graph.png", dpi=200)
    plt.clf()

# Saves period-angle graph
def powerSeries(angle, initialTime, B, C):
    return initialTime*(1+B*angle+C*angle**2)

periodAngleData = {} # length: [xValues, yValues, x uncertainties, y uncertainties]
for index in means:
    if index[0] in periodAngleData:
        periodAngleData[index[0]][0].append(index[1])
        periodAngleData[index[0]][1].append(means[index][0])
        periodAngleData[index[0]][2].append(initialAngleUncertainty)
        periodAngleData[index[0]][3].append(uncertaintyTotal[index][0])
    else:
        periodAngleData[index[0]] = [[index[1]], [means[index][0]], [initialAngleUncertainty], [uncertaintyTotal[index][0]]]

for length in periodAngleData:
    xValues, yValues, xUncertainties, yUncertainties = periodAngleData[length]

    if len(xValues) <= 10:
        continue

    params, covariance = curveFit(powerSeries, xValues, yValues)
    initialTime, B, C = params
    initialTimeUnc, bUnc, cUnc = np.sqrt(np.diagonal(covariance))

    print("Graph Name: "+"-".join(str(float(length)).split("."))+"-graph.png")
    print("Line of best fit parameters for T = Initial Value * (1 + B*θ + C*θ^2)")
    print("Initial Value:", roundUncertainties(initialTime, initialTimeUnc))
    print("B Value:", roundUncertainties(B, bUnc))
    print("C Value:", roundUncertainties(C, cUnc))
    print("R^2 Value:", r2Score(yValues, np.array([powerSeries(x, initialTime, B, C) for x in xValues])))
    print("Control Variable (Length [cm]):", roundUncertainties(length, lengthUncertainty))
    print("\n\n")
    
    plt.errorbar(xValues, yValues, xerr = xUncertainties, yerr = yUncertainties, fmt='o')
    plt.xlabel("Initial Angle [rad]", fontsize = 12)
    plt.ylabel("Period [s]", fontsize = 12)

    plt.plot(np.arange(min(xValues), max(xValues), 0.001), [powerSeries(x, initialTime, B, C) for x in np.arange(min(xValues), max(xValues), 0.001)])
    fig = plt.gcf()
    fig.set_size_inches(10, 7)
    fig.savefig(periodFolderPath+"period-angle-graphs\\period-angle-"+"-".join(str(float(length)).split("."))+"-graph.png", dpi=200)
    plt.clf()


# Saves q-factor-length graph
def qFactorFitLength(length, m, b):
    return m*length+b

qLengthData = {} # angle: [xValues, yValues, x uncertainties, y uncertainties]
qLengthData2 = {} # angle: [xValues, yValues, x uncertainties, y uncertainties]
for index in means:
    if index[1] in qLengthData:
        qLengthData[index[1]][0].append(index[1])
        qLengthData[index[1]][1].append(means[index][2])
        qLengthData[index[1]][2].append(lengthUncertainty)
        qLengthData[index[1]][3].append(uncertaintyTotal[index][2])

        qLengthData2[index[1]][0].append(index[1])
        qLengthData2[index[1]][1].append(means[index][3])
        qLengthData2[index[1]][2].append(lengthUncertainty)
        qLengthData2[index[1]][3].append(uncertaintyTotal[index][3])
    else:
        qLengthData[index[1]] = [[index[0]], [means[index][2]], [lengthUncertainty], [uncertaintyTotal[index][2]]]
        qLengthData2[index[1]] = [[index[0]], [means[index][3]], [lengthUncertainty], [uncertaintyTotal[index][3]]]

for angle in qLengthData:
    xValues, yValues, xUncertainties, yUncertainties = qLengthData[angle]

    if len(xValues) <= 3:
        continue

    params, covariance = curveFit(qFactorFitLength, xValues, yValues)
    m, b = params
    mUnc, bUnc = np.sqrt(np.diagonal(covariance))

    print("Graph Name: "+"-".join(str(float(length)).split("."))+"-graph.png")
    print("Line of best fit parameters for Q = m * Length + b")
    print("m Value:", roundUncertainties(m, mUnc))
    print("b Value:", roundUncertainties(b, bUnc))
    print("R^2 Value:", r2Score(yValues, np.array([qFactorFitLength(x, m, b) for x in xValues])))
    print("Control Variable (Initial Angle [rad]):", roundUncertainties(angle, initialAngleUncertainty))
    print("\n\n")
    
    plt.errorbar(xValues, yValues, xerr = xUncertainties, yerr = yUncertainties, fmt='o')
    plt.xlabel("Lengths [cm]", fontsize = 12)
    plt.ylabel("Q Factor [unitless]", fontsize = 12)

    plt.plot(np.arange(min(xValues), max(xValues), 0.001), [qFactorFitLength(x, m, b) for x in np.arange(min(xValues), max(xValues), 0.001)])
    fig = plt.gcf()
    fig.set_size_inches(10, 7)
    fig.savefig(qFactorFolderPath+"q-factor-length-graphs\\q-factor-length-"+"-".join(str(float(angle)).split("."))+"-graph.png", dpi=200)
    plt.clf()

# Saves q-factor-angle graph
qAngleData = {} # length: [xValues, yValues, x uncertainties, y uncertainties]
qAngleData2 = {} # length: [xValues, yValues, x uncertainties, y uncertainties]
for index in means:
    if index[0] in qAngleData:
        qAngleData[index[0]][0].append(index[1])
        qAngleData[index[0]][1].append(means[index][2])
        qAngleData[index[0]][2].append(initialAngleUncertainty)
        qAngleData[index[0]][3].append(uncertaintyTotal[index][2])
        
        qAngleData2[index[0]][0].append(index[1])
        qAngleData2[index[0]][1].append(means[index][3])
        qAngleData2[index[0]][2].append(initialAngleUncertainty)
        qAngleData2[index[0]][3].append(uncertaintyTotal[index][3])
    else:
        qAngleData[index[0]] = [[index[1]], [means[index][2]], [initialAngleUncertainty], [uncertaintyTotal[index][2]]]
        qAngleData2[index[0]] = [[index[1]], [means[index][3]], [initialAngleUncertainty], [uncertaintyTotal[index][3]]]

for length in qAngleData:
    xValues, yValues, xUncertainties, yUncertainties = qAngleData[length]

    if len(xValues) <= 10:
        continue

    print("Graph Name: "+"-".join(str(float(length)).split("."))+"-graph.png")
    print("Control Variable (Length [cm]):", roundUncertainties(length, lengthUncertainty))
    print("\n\n")
    
    plt.errorbar(xValues, yValues, xerr = xUncertainties, yerr = yUncertainties, fmt='o', label="Method 1", ms=1)
    xValues, yValues, xUncertainties, yUncertainties = qAngleData2[length]
    plt.errorbar(xValues, yValues, xerr = xUncertainties, yerr = yUncertainties, fmt='o', label="Method 2", ms=1)
    plt.xlabel("Initial Angle [rad]", fontsize = 12)
    plt.ylabel("Q Factor [unitless]", fontsize = 12)
    plt.legend(loc="upper left")

    fig = plt.gcf()
    fig.set_size_inches(10, 7)
    fig.savefig(qFactorFolderPath+"q-factor-angle-graphs\\q-factor-angle-"+"-".join(str(float(length)).split("."))+"-graph.png", dpi=200)
    plt.clf()