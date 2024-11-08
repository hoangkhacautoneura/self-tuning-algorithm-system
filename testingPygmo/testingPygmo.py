import pygmo as pg
import numpy as np
import argparse
import pandas as pd
# Step 1: Define a custom problem class for x^2
class MyProblem:
    def __init__(self, dataFrame):
        self.dim = 3  # The problem is 1-dimensional
        self.setPointTrace = dataFrame['sp']
        self.response = dataFrame['response']
        print(self.setPointTrace)
    # Fitness function: the objective we are minimizing (x^2)
    def fitness(self, x):
        # Kp = x[0]
        # Ki = x[1]
        # Kd = x[2]
        # PIDParameters = [x[0], x[1], x[2]]
        # PIDParameters = [1,2,3]
        # print(f'PIDParameters outside: {PIDParameters}')
        ISE, Tr, Mp, Ts, Ess = computePIDCriterias(x, self.setPointTrace, self.response)
        # print(f"ISE: {ISE}")
        # print(f"Tr (Rise Time): {Tr}")
        # print(f"Mp (Maximum Overshoot): {Mp}")
        # print(f"Ts (Settling Time): {Ts}")
        # print(f"Ess (Steady-State Error): {Ess}")
        return  [0.15*ISE + 0.4*Tr + 0.05*Mp + 0.2*Ts + 0.2*Ess]  # The function to minimize is x^2
    
    # Get bounds for the decision variable (range for x)
    def get_bounds(self):
        return ([0, 0, 0], [30, 30, 30])  # x can range between -10 and 10

def objective_function(x, y, z):
    return (x**2 + y**2 + z**2) + np.sin(5 * x) * np.sin(5 * y) * np.sin(5 * z)

def computePIDCriterias(PIDParameters, setPointTrace, response):
    # print(f'PIDParameters inside: {PIDParameters}')
    measuredSpeed = performPIDSimulation(setPointTrace, PIDParameters)
    # Computing integral squared error (ISE)
    # measuredSpeed = response
    ISE = sum((a - b) ** 2 for a, b in zip(setPointTrace, measuredSpeed))

    changeIndex = 0
    setPointTraceStepValue = 0
    for i in range(1, len(setPointTrace)):
        if(setPointTrace[i] != setPointTrace[i-1]):
            changeIndex = i
            setPointTraceStepValue = setPointTrace[i]
            break;
    # Compute rise time
    riseTimeStart = 0
    riseTimeEnd = 0
    riseTime = 0
    for i in range(changeIndex, len(measuredSpeed)):
        if(measuredSpeed[i] > setPointTraceStepValue*10/100):
            riseTimeStart = i
        if(measuredSpeed[i] > setPointTraceStepValue * 90/100):
            riseTimeEnd = i
            riseTime = riseTimeEnd - riseTimeStart
            break;
    # compute maximum overshoot percentage
    maximumOvershootPercentage = (max(measuredSpeed) - setPointTraceStepValue)/setPointTraceStepValue
    
    # compute settiling time, 2% tolerance
    tolerance = 2/100;
    upperSettledThreshold = (1+tolerance)*setPointTraceStepValue
    lowerSettledThreshold = (1-tolerance)*setPointTraceStepValue
    # print(f'uppserSettledThreshold is: {upperSettledThreshold}')
    # print(f'lowerSettledThreshold is: {lowerSettledThreshold}')
    settlingTime = 0;
    systemSettled = False;
    for i in range(changeIndex, len(measuredSpeed)):
        speedWithinRange = (measuredSpeed[i] < upperSettledThreshold) and (measuredSpeed[i] > lowerSettledThreshold)
        if(speedWithinRange and (systemSettled == False)):
            settlingTime = i
            # print(f'measured Speed is: {measuredSpeed[i]}')
            # print(f'settling time is: {settlingTime}')
            systemSettled = True
        if( not speedWithinRange) :
            systemSettled = False

    # compute steady state error
    steadyStateError = 0
    for i in range(settlingTime, len(measuredSpeed)):
        steadyStateError += abs(setPointTrace[i] - measuredSpeed[i])
    return [ISE, riseTime, maximumOvershootPercentage, settlingTime, steadyStateError]

def performPIDSimulation(setPoint, PIDParameters):
    currentSpeed = 0
    sumError = 0
    previousError = 0
    measuredSpeed = []
    # print(f'Kp: {PIDParameters[0]}')
    # print(f'Ki: {PIDParameters[1]}')
    # print(f'Kd: {PIDParameters[2]}')
    for i in range(0, len(setPoint)):
        error = setPoint[i] - currentSpeed
        sumError += error
        pedalPosition = PIDParameters[0]*error + PIDParameters[1]*sumError + PIDParameters[2]*(error - previousError)
        previousError = error
        currentSpeed = pedalPosition*0.01
        # print(f'error: {error}')
        # print(f'currentSpeed: {currentSpeed}')
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

# ISE, Tr, Mp, Ts, Ess = computePIDCriterias(0, df['sp'], df['response'])

# print(f"ISE: {ISE}")
# print(f"Tr (Rise Time): {Tr}")
# print(f"Mp (Maximum Overshoot): {Mp}")
# print(f"Ts (Settling Time): {Ts}")
# print(f"Ess (Steady-State Error): {Ess}")
fail = 0
success = 0
# for i in range(50):
# problem = pg.problem(MyProblem(df))

# algo = pg.algorithm(pg.pso(gen=10, omega=0.9,max_vel = 1))
# algo.set_verbosity(1)
# Step 4: Create a population (size 10)
# pop = pg.population(problem, size=10)

# pop = algo.evolve(pop)
# print("Best solution found: x =", pop.champion_x)
# print("Minimum value of x^2: f(x) =", pop.champion_f)

# Step 5: Evolve the population to solve the problem
previousChampionF = 0;
generation = 10
# for i in range(generation):
#     if(pop.champion_f != previousChampionF):
#         measureSpeed = performPIDSimulation(df['sp'], pop.champion_x)
#         dataFrame = pd.DataFrame(measureSpeed, df['sp'])
#         stringName = 'measuredSpeed' + str(i) + '.csv'
#         dataFrame.to_csv(stringName, index=True)
#         print("Best solution found: x =", pop.champion_x)
#         print("Minimum value of x^2: f(x) =", pop.champion_f)
#     previousChampionF = pop.champion_f
averageSuccessGBest = 0;
for i in range(1):
    problem = pg.problem(MyProblem(df))
    algo = pg.algorithm(pg.pso(gen=20, omega=0.9,max_vel = 1))
    # algo.set_verbosity(1)
    # Step 4: Create a population (size 10)
    pop = pg.population(problem, size=20)
    pop = algo.evolve(pop)
    measureSpeed = performPIDSimulation(df['sp'], pop.champion_x)
    dataFrame = pd.DataFrame(measureSpeed, df['sp'])
    stringName = 'measuredSpeed' + str(i) + '.csv'
    dataFrame.to_csv(stringName, index=True)
    if(pop.champion_f > 100000):
        fail += 1
    else:
        success += 1
        averageSuccessGBest += pop.champion_f

print(f'fail is: {fail}')
print(f'success is:{success}')
print(f'averageSuccessGBest is:{averageSuccessGBest/100}')

print("Best solution found: x =", pop.champion_x)
print("Minimum value of x^2: f(x) =", pop.champion_f)
# Step 6: Output the results