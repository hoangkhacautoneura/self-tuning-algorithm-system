import pygmo as pg
import numpy as np
import argparse
import pandas as pd
import math
from PIDSimulation import PIDSimulation
# Step 1: Define a custom problem class for x^2
class MyProblem:
    globalHistory = []

    def __init__(self, dataFrame, setPointSlopes, curveArea,  setPointInflectionPoints, curveDifferentialDistance, inflectionPointSearchDistance):
        self.dim = 3  # The problem is 1-dimensional
        self.dataFrame = dataFrame
        self.setPointTrace = dataFrame['speed']
        self.curveArea = curveArea
        # self.testResponse = dataFrame['response']
        self.setPointSlopes = setPointSlopes
        self.setPointInflectionPoints = setPointInflectionPoints
        self.timeStep = dataFrame['time'][1] - dataFrame['time'][0]
        self.curveDifferentialDistance = curveDifferentialDistance
        self.inflectionPointSearchDistance = inflectionPointSearchDistance
        self.history = []
        # self.response = dataFrame['response']
        # self.computePIDCriterias([0,0,0], self.testResponse)
        # print(self.setPointTrace)
    # Fitness function: the objective we are minimizing (x^2)
    def fitness(self, x):
        # Kp = x[0]
        # Ki = x[1]
        # Kd = x[2]
        # PIDParameters = [x[0], x[1], x[2]]
        # PIDParameters = [1,2,3]
        # print(f'PIDParameters outside: {PIDParameters}')
        ISE, slopeError, slopeInflectionPoints, totalCurveDifferentialError, inflectionPointMagnitudeError, timeError = self.computePIDCriterias(x, [])
        print("computed the error")
        totalSlopeError = sum(slopeError)
        numberOfSlopeInflectionPoints = 0
        totalInflectionPointTimeError = 0
        totalMagnitude = 0
        for i in slopeInflectionPoints:
            numberOfSlopeInflectionPoints += i[0]
            totalMagnitude += i[1]
        for i in timeError:
            totalInflectionPointTimeError += i
        # print(f"ISE: {ISE}")
        # print(f"Tr (Rise Time): {Tr}")
        # print(f"Mp (Maximum Overshoot): {Mp}")
        # print(f"Ts (Settling Time): {Ts}")
        # print(f"Ess (Steady-State Error): {Ess}")
        MyProblem.globalHistory.append({
            'ISE': ISE,
            'totalSlopeError' : totalSlopeError,
            'numberOfSlopeInflectionPoints' : numberOfSlopeInflectionPoints,
            'totalMagnitude' : totalMagnitude,
            'totalCurveDifferentialError' : totalCurveDifferentialError,
            'inflectionPointMagnitudeError' : inflectionPointMagnitudeError,
            'totalInflectionPointTimeError' : totalInflectionPointTimeError,
        })
        self.ISE = ISE
        self.totalSlopeError = totalSlopeError
        self.numberOfSlopeInflectionPoints = numberOfSlopeInflectionPoints
        self.totalMagnitude = totalMagnitude
        self.totalCurveDifferentialError = totalCurveDifferentialError
        self.inflectionPointMagnitudeError = inflectionPointMagnitudeError
        self.totalInflectionPointTimeError = totalInflectionPointTimeError
        # return [ISE + totalSlopeError + numberOfSlopeInflectionPoints + totalMagnitude + totalCurveDifferentialError + inflectionPointMagnitudeError + totalInflectionPointTimeError]
        return [ISE]
    
    def getCriteriaValues(self):
        print(MyProblem.globalHistory)
        return MyProblem.globalHistory

    # Get bounds for the decision variable (range for x)
    def get_bounds(self):
        return ([0, 0, 0], [0.1, 0.1, 0.1])  # x can range between -10 and 10
    
    def computePIDCriterias(self, PIDParameters, testResponse):
        # print(f'PIDParameters inside: {PIDParameters}')
        
        measuredSpeed = PIDSimulation(PIDParameters, self.dataFrame)
        # measuredSpeed = testResponse
        
        # Computing integral squared error (ISE)
        ISE = sum((a - b) ** 2 for a, b in zip(self.setPointTrace, measuredSpeed))
        slopeError, slopeInflectionPoints = self.computeSlopeError(measuredSpeed)
        totalCurveDifferentialError = self.computeCurveError(measuredSpeed)
        magnitudeError, timeError = self.computeInflectionPointError(measuredSpeed)
        
        # Computing errors at slope
        # print("ISE :", ISE)
        # print("Slope Error:", slopeError)
        # print("Slope Inflection Points:", slopeInflectionPoints)
        # print("Total Curve Differential Error:", totalCurveDifferentialError)
        # print("Magnitude Error:", magnitudeError)
        # print("Time Error:", timeError)
        return [ISE, slopeError, slopeInflectionPoints, totalCurveDifferentialError, magnitudeError, timeError]
    
    def computeSlopeError(self, measuredSpeed):
        slopeError =[]
        slopeInflectionPoints = [];
        for i in range(0, len(self.setPointSlopes)):
            startArrayPosition = int(self.setPointSlopes[i][0]/self.timeStep)
            endArrayPosition = int(self.setPointSlopes[i][1]/self.timeStep)

            # y = mx + c
            m = self.setPointSlopes[i][2]
            c = self.setPointSlopes[i][3]
            measuredSlope = (measuredSpeed[endArrayPosition] - measuredSpeed[startArrayPosition])/( (endArrayPosition - startArrayPosition)*self.timeStep)
            # print('startArrayPosition', startArrayPosition)
            # print('endArrayPosition', endArrayPosition)
            # print('measuredSlope', measuredSlope)
            
            slopeError.append(abs(m - measuredSlope))

            perpendicularDistanceRange = []
            verticalDistanceList = []
            for j in range(startArrayPosition, endArrayPosition):
                yOnLine = m *(j - startArrayPosition + 1)*self.timeStep + c
                # print("measuredSpeed[j]", measuredSpeed[j])
                # print("yOnLine", yOnLine)
                # Calculate the vertical distance
                verticalDistance = measuredSpeed[j] - yOnLine
                # Calculate the perpendicular distance
                perpendicularDistanceRange.append(verticalDistance / math.sqrt(1 + m**2))
                verticalDistanceList.append(verticalDistance)

            inflectionPointCount = 0
            inflectionPointDistance = 0

            for i in range(1, len(perpendicularDistanceRange) - 1):
                if((perpendicularDistanceRange[i] - perpendicularDistanceRange[i-1] > 0 and perpendicularDistanceRange[i+1] - perpendicularDistanceRange[i] < 0) or 
                   (perpendicularDistanceRange[i] - perpendicularDistanceRange[i-1] < 0 and perpendicularDistanceRange[i+1] - perpendicularDistanceRange[i] > 0)):
                    # print("perpendicularDistanceRange[i-1]", perpendicularDistanceRange[i-1])
                    # print("perpendicularDistanceRange[i]", perpendicularDistanceRange[i])
                    # print("perpendicularDistanceRange[i+1]", perpendicularDistanceRange[i+1])
                    # print("verticalDistanceList[i]", verticalDistanceList[i])
                    # print("verticalDistanceList[i+1]", verticalDistanceList[i+1])
                    # print("verticalDistanceList[i-1]", verticalDistanceList[i-1])
                    inflectionPointCount+=1
                    inflectionPointDistance += abs(perpendicularDistanceRange[i])
            slopeInflectionPoints.append([inflectionPointCount, inflectionPointDistance])
        return slopeError, slopeInflectionPoints
    
    def computeCurveError(self, measuredSpeed):
        setPointSecondOrderDifferential = self.computeCurveSecondOrderDifferential(self.setPointTrace)
        measureSpeedSecondOrderDifferential = self.computeCurveSecondOrderDifferential(measuredSpeed)
        secondOrderDifferentialDifference = []
        for i in range(0, len(setPointSecondOrderDifferential)):
            # print("SetPoint: ", setPointSecondOrderDifferential[i])
            # print("Measrued: ", measureSpeedSecondOrderDifferential[i])
            setPointMeasureDifferece = abs(setPointSecondOrderDifferential[i] - measureSpeedSecondOrderDifferential[i])
            secondOrderDifferentialDifference.append(setPointMeasureDifferece)
            # if abs(setPointSecondOrderDifferential[i] - measureSpeedSecondOrderDifferential[i]) > 0:
                # print("index is: ", i)
                # print("difference is: ", setPointSecondOrderDifferential[i] - measureSpeedSecondOrderDifferential[i])
        # for i in range(0, len(secondOrderDifferentialDifference)):
        #     print("secondOrderDifferentialDifference: ", i)
        #     print(secondOrderDifferentialDifference[i])
        totalCurveDifferentialError = sum(secondOrderDifferentialDifference)
        return totalCurveDifferentialError

    def computeCurveSecondOrderDifferential(self, trace):
        secondOrderDifferentialCurveError = []
        for i in range(0, len(self.curveArea)):
        # print(CurveStart)
        # print(len(trace))
            for j in range(int(self.curveArea[i][0]*10), int(self.curveArea[i][1]*10 + 1)):
                if(j != len(trace) - 1):
                    currentPointDifferential = (trace[j] - trace[j-self.curveDifferentialDistance])/(self.timeStep*self.curveDifferentialDistance)
                    previousPointDifferential = (trace[j-1] - trace[j-1-self.curveDifferentialDistance])/(self.timeStep*self.curveDifferentialDistance)
                    secondOrderDifferentialCurveError.append((currentPointDifferential - previousPointDifferential)/(self.timeStep*self.curveDifferentialDistance))
                    # if i == CurveStart:
                        # print("abcd")
                        # print("CurveStart", CurveStart)
                        # print("trace[i]", trace[i])
                        # print("currentPointDifferential", currentPointDifferential)
                        # print("previousPointDifferential", previousPointDifferential)
        return secondOrderDifferentialCurveError

    def computeInflectionPointError(self, measuredSpeed):
        magnitudeError = 0
        timeError = []
        for i in range(0, len(self.setPointInflectionPoints)):
            arrayIndex = int(self.setPointInflectionPoints[i][0]/self.timeStep)
            magnitudeError += abs(self.setPointTrace[arrayIndex] - measuredSpeed[arrayIndex])
            inflectionPointFound = False
            # print(self.setPointInflectionPoints[i][2])
            # print("arrayIndex", arrayIndex)
            measuredSpeedInflectionPoints = []
            for j in range(arrayIndex - self.inflectionPointSearchDistance, arrayIndex + self.inflectionPointSearchDistance):
                if(self.setPointInflectionPoints[i][2] == 'peak'):
                    if((measuredSpeed[j] - measuredSpeed[j-1] > 0) and (measuredSpeed[j+1] - measuredSpeed[j] < 0) ):
                        measuredSpeedInflectionPoints.append(j)
                        inflectionPointFound = True
                if(self.setPointInflectionPoints[i][2] == 'trough'):
                    if((measuredSpeed[j] - measuredSpeed[j-1] < 0) and (measuredSpeed[j+1] - measuredSpeed[j] > 0) ):
                        measuredSpeedInflectionPoints.append(j)
                        inflectionPointFound = True
            closestInflectionPoint = 0
            # print("infelction points length: ", measuredSpeedInflectionPoints)
            if(inflectionPointFound):
                closestInflectionPoint = min(abs(x - arrayIndex) for x in measuredSpeedInflectionPoints)
            else:
                closestInflectionPoint = self.inflectionPointSearchDistance
            # print("closestInflectionPoint", closestInflectionPoint)
            timeError.append(closestInflectionPoint*self.timeStep)
        return magnitudeError, timeError

def performPIDSimulation(setPoint, PIDParameters):
    currentSpeed = 0
    sumError = 0
    previousError = 0
    previousSpeed = 0
    measuredSpeed = []
    # print(f'Kp: {PIDParameters[0]}')
    # print(f'Ki: {PIDParameters[1]}')
    # print(f'Kd: {PIDParameters[2]}')
    for i in range(0, len(setPoint)):
        error = setPoint[i] - currentSpeed
        sumError += error
        pedalPosition = PIDParameters[0]*error + PIDParameters[1]*sumError + PIDParameters[2]*(error - previousError)
        previousError = error
        # currentSpeed = (pedalPosition*0.01 + previousSpeed)/2
        currentSpeed = (pedalPosition*0.01 + previousSpeed)/2
        # print(f'error: {error}')
        # print(f'currentSpeed: {currentSpeed}')
        previousSpeed = currentSpeed
        measuredSpeed.append(currentSpeed)
    return measuredSpeed

# Initialize the argument parser
parser = argparse.ArgumentParser(description="Input a .csv file")

# Add an argument for the CSV file path
parser.add_argument('csv_file', type=str, help="Path to the .csv file")

# Parse the arguments
args = parser.parse_args()

# Read the CSV file using pandas
df = pd.read_csv(args.csv_file)

timeMultiplication = 500
# y=mx+c
# set point slope for SetPointTraceTestingSimple
setPointSlope = [[0,3,0,1500], [3.1,6,0.8*timeMultiplication, 1500], [6.1,10,0.4*timeMultiplication, 2700], [10.1,14,-0.3*timeMultiplication, 3500], [14.1, 17, 0, 2900], [17.1,19.5,-0.7*timeMultiplication, 2900], [19.6,24,0.8*timeMultiplication, 2025]]
setPointInflectionPoint = [[10, 3500, 'peak'], [19.5, 2025, 'trough']]
curveDifferentialDistance = 5
curveArea = []

#set point slope for WLTC_single_bum
# setPointSlope = [[0,6.9,0, 750], [7,8,440.2032,938.1179848],[8.1, 248.3581, 1378.321], [13.1,19, -176.76, 1801.467148]]
# setPointInflectionPoint = [[11.3, 1918.808, 'peak']]
# curveDifferentialDistance = 5;
# curveArea = [[10.1,13]]

inflectionPointSearchDistance = 10;
fail = 0
success = 0
# Step 5: Evolve the population to solve the problem
previousChampionF = 0
generation = 10
populationSize = 10
repetition = 1
averageSuccessGBest = 0
averageISE = 0
averageSlopeError = 0
averageNumberOfSlopeInflectionPoints = 0
averageInflectionPointMagnitudeError = 0
averageTotalInflectionPointTimeError = 0
averageTotalCurveDifferentialError = 0
print("working")
for i in range(repetition):
    # Set up problem and solve
    problem = pg.problem(MyProblem(df, setPointSlope, curveArea, setPointInflectionPoint, curveDifferentialDistance, inflectionPointSearchDistance))
    algo = pg.algorithm(pg.pso(gen=generation))
    # algo.set_verbosity(1)
    pop = pg.population(problem, size=populationSize)
    pop = algo.evolve(pop)
    print("Paramters are: ",pop.champion_x)
    # # Look into the criteria individual values
    computedHistory = problem.extract(MyProblem).getCriteriaValues()
    print(computedHistory)
    CHDataFrame = pd.DataFrame(computedHistory)
    csvFileName = 'result/criteriaValues.csv'
    CHDataFrame.to_csv(csvFileName, index=True)

    # Output into CSV
    measureSpeed = PIDSimulation(pop.champion_x, df)
    dataFrame = pd.DataFrame(measureSpeed, df['speed'])
    stringName = 'result/PSOSimulationG10P10' + str() + '.csv'
    dataFrame.to_csv(stringName, index=True)
    
    # # Extracting individual objectives output according to the parameters we have found
    # testingProblem = MyProblem(df, setPointSlope, setPointInflectionPoint, curveDifferentialDistance, inflectionPointSearchDistance)
    # ISE, slopeError, slopeInflectionPoints, totalCurveDifferentialError, magnitudeError, timeError = testingProblem.computePIDCriterias(pop.champion_x, [])
    # totalSlopeError = sum(slopeError)

    # totalMagnitude = 0    
    # numberOfSlopeInflectionPoints = 0
    # totalInflectionPointTimeError = 0
    # for i in slopeInflectionPoints:
    #     numberOfSlopeInflectionPoints += i[0]
    #     totalMagnitude += i[1]
    # for i in timeError:
    #     totalInflectionPointTimeError += i

    # averageISE += ISE
    # averageSlopeError += totalSlopeError
    # averageNumberOfSlopeInflectionPoints += numberOfSlopeInflectionPoints
    # averageInflectionPointMagnitudeError += totalMagnitude
    # averageTotalInflectionPointTimeError += totalInflectionPointTimeError
    # averageTotalCurveDifferentialError += totalCurveDifferentialError
    
    # if(pop.champion_f > 100000):
    #     fail += 1
    # else:
    #     success += 1
    #     averageSuccessGBest += pop.champion_f

print("Average ISE:", averageISE/repetition)
print("Average Slope Error:", averageSlopeError/repetition)
print("Average Number of Slope Inflection Points:", averageNumberOfSlopeInflectionPoints/repetition)
print("Average Inflection Point Magnitude Error:", averageInflectionPointMagnitudeError/repetition)
print("Average Total Inflection Point Time Error:", averageTotalInflectionPointTimeError/repetition)
print("Average Total Curve Differential Error:", averageTotalCurveDifferentialError/repetition)

print("Paramters are: ",pop.champion_x)
# print("ISE :", ISE)
# print("TSE:", totalSlopeError)
# print("NSIP: ", numberOfSlopeInflectionPoints)
# print("TIME:", magnitudeError)
# print("TITE: ", totalInflectionPointTimeError)
# print("TCDE:", totalCurveDifferentialError)
# print("Slope Inflection Points:", slopeInflectionPoints)
# print("Time Error:", timeError)
# print("Individual slope error:", slopeError)

# print(f'fail is: {fail}')
# print(f'success is:{success}')
# print(f'averageSuccessGBest is:{averageSuccessGBest/repetition}')

# print("Best solution found: x =", pop.champion_x)
# print("Minimum value of x^2: f(x) =", pop.champion_f)
# Step 6: Output the results