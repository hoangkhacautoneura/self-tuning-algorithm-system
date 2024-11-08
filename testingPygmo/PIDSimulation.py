from datetime import time
import numpy as np
import pandas as pd
import statistics
from tensorflow import keras
import csv
Model = keras.Model
layers = keras.layers
preprocessing = keras.preprocessing
optimizers = keras.optimizers
callbacks = keras.callbacks
models = keras.models

# import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# GPU CUDA Enablement
# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

def computeMean():
    mean = 0

def bilinear_interpolation(x,y,points):
    # need bi-linear interpolation response map source: https://en.wikipedia.org/wiki/Bilinear_interpolation
    # Scipy could work but needs splitting of the array - interpolate.interp2d(x, y, z, kind='linear')

    
    # Feedforward Map Definition
    # ------------
    # FEEDFORWARD MAP #1 BASELINE PEDAL POSITION
    # Note: For pedal positions which are not attainable through testing, pedal positions are linearly interpolated

    feedforward_baseline_np = [[0.025, 0.6625, 1.3, 1.9375, 2.575, 3.2125, 3.85, 4.4875, 5.125, 5.7625, 6.4, 7.0375, 7.676, 8.3125, 8.95, 9.6385, 10.225, 10.8625, 11.5, 12.1375],
    [7.9528, 8.8403, 9.7278, 10.6153, 11.5028, 12.3903, 13.2778, 14.1653, 15.0528, 15.9403, 16.8278, 17.7153, 18.6028, 19.4903, 20.3778, 21.3363, 22.1528, 23.0403, 23.9278, 24.8153],
    [16.232, 17.1695, 18.107, 19.0445, 19.982, 20.9195, 21.857, 22.7945, 23.732, 24.6695, 25.607, 26.5445, 27.482, 28.4195, 29.357, 30.3695, 31.232, 32.1695, 33.107, 34.0445],
    [24.725, 25.5125, 26.3, 27.0875, 27.875, 28.6625, 29.45, 30.2375, 31.025, 31.8125, 32.6, 33.3875, 34.175, 34.9625, 35.75, 36.6005, 37.325, 38.1125, 38.9, 39.6875],
    [30.146, 31.1085, 32.071, 33.0335, 33.996, 34.9585, 35.921, 36.8835, 37.846, 38.8085, 39.771, 40.7335, 41.696, 42.6585, 43.621, 44.6605, 45.546, 46.5085, 47.471, 48.4335],
    [39.133, 40.333, 41.533, 42.733, 43.933, 45.133, 46.333, 47.533, 48.733, 49.933, 51.133, 52.33, 54.733, 55.933, 57.229, 58.333, 59.333, 60.733, 61.933, 65.15],
    [55.8, 56.45, 57.1, 57.75, 58.4, 59.05, 59.7, 60.35, 61, 61.65, 62.3, 62.95, 63.6, 64.25, 64.9, 65.602, 66.2, 66.85, 67.5, 68.15]]    


    # Note: Converting array to pandas dataframe with columna and index naming
    feedforward_baseline_df = pd.DataFrame(feedforward_baseline_np, columns=['750', '875', '1000', '1125', '1250', '1375', '1500', '1625', '1750', '1875', '2000', '2125', '2250', '2375', '2500', '2635', '2750', '2875', '3000', '3125'], index=['4.9','50','100', '150', '200', '250', '300'])
    # ------------
    # Step 2 - END

    if (points[0] < 4.9):
        points_0 = 4.9
    else:
        points_0 = points[0]
    if (points[1] < 750):
        points_1 = 751
    else:
        points_1 = points[1]

    #print('Points 0', points_0)
    #print('Points 1', points_1)

    for column_cell in range(0,len(y)): 
        if float(points_1) >= y[column_cell] and float(points_1) <= y[column_cell+1]:
            upper_column_cell = y[column_cell+1]
            lower_column_cell = y[column_cell]
    for row_cell in range(0,len(x)): 
        #print('x row cell', x[row_cell])
        if float(points_0) >= x[row_cell] and float(points_0) <= x[row_cell+1]:
            upper_row_cell = x[row_cell+1]
            lower_row_cell = x[row_cell]

    #print('upper speed cell', upper_column_cell)
    #print('lower speed cell', lower_column_cell)
    #print('upper torque cell', upper_row_cell)
    #print('lower torque cell', lower_row_cell)            

    lhs_initial = ((upper_column_cell-points[1])/(upper_column_cell-lower_column_cell))*feedforward_baseline_df.loc[str(lower_row_cell), str(lower_column_cell)] + ((points[1]-lower_column_cell)/(upper_column_cell-lower_column_cell))*feedforward_baseline_df.loc[str(lower_row_cell), str(upper_column_cell)]
    rhs_initial = ((upper_column_cell-points[1])/(upper_column_cell-lower_column_cell))*feedforward_baseline_df.loc[str(upper_row_cell), str(lower_column_cell)] + ((points[1]-lower_column_cell)/(upper_column_cell-lower_column_cell))*feedforward_baseline_df.loc[str(upper_row_cell), str(upper_column_cell)]
    lhs = ((upper_row_cell - points[0])/(upper_row_cell-lower_row_cell))*lhs_initial
    rhs = ((points[0] - lower_row_cell)/(upper_row_cell-lower_row_cell))*rhs_initial

    interpolated_value = max(1.6, lhs + rhs)

    #print('Interpolated Value', interpolated_value)
    
    return interpolated_value


# columns = ['Target_N', 'Target_Tb', 'pedal', 'dN_prev', 'dN', 'dN_Next', 'dT_Prev', 'dT', 'dT_Next', 'Error_Prev' ]
def extractFromDataFrameInput(input):
    targetNArray = []
    targetTbArray = []
    dNPrevArray = []
    dNArray = []
    dnNextArray = []
    dTPrevArray = []
    dtArray = []
    dTNextArray = []
    for row in range(0, len(input)):
        if (row == 0):
            targetNArray.append(input['speed'][row]) 
            targetTbArray.append(input['torque'][row])
            dNPrevArray.append(0)
            dNArray.append(0)
            dnNextArray.append(input['speed'][row+1] - input['speed'][row])
            dTPrevArray.append(0)
            dtArray.append(0)
            dTNextArray.append(input['torque'][row+1] - input['torque'][row])
        else:
            targetNArray.append(input['speed'][row])
            targetTbArray.append(input['torque'][row])

            prev_target_Tb = input['torque'][row - 1]
            prev_target_N = input['speed'][row - 1]

            # Ending values for next targets
            if (row == len(input) - 1):
                nxt_target_Tb = 0
                nxt_target_N = 0
            else:
                nxt_target_Tb = input['torque'][row + 1]
                nxt_target_N = input['speed'][row + 1]

            # Initial values for 
            if (row > 1):
                prev_two_target_N = input['speed'][row - 2]
                prev_two_target_Tb = input['torque'][row - 2]
            else: 
                prev_two_target_N = 0
                prev_two_target_Tb = 0
            
            # LSTM Inputs
            dNPrevArray.append(prev_target_N - prev_two_target_N)
            # derivative_N_prev = prev_target_N - prev_two_target_N
            dNArray.append(targetNArray[row] - prev_target_N)
            # derivative_N = target_N - prev_target_N
            dnNextArray.append(nxt_target_N - prev_target_N)
            # derivative_N_next = nxt_target_N - prev_target_N
            dTPrevArray.append(prev_target_Tb - prev_two_target_Tb)
            # derivative_T_prev = prev_target_Tb - prev_two_target_Tb
            dtArray.append(targetTbArray[row] - prev_target_Tb)
            # derivative_T = target_Tb - prev_target_Tb
            dTNextArray.append(nxt_target_Tb - prev_target_Tb)
            # derivative_T_next = nxt_target_Tb - prev_target_Tb

    dataMeanArray, dataStdevArray = computeDataMeanAndStdev(targetNArray, targetTbArray, dNPrevArray, dNArray, dnNextArray, dTPrevArray, dtArray, dTNextArray)

    # checking = np.column_stack((targetNArray, targetTbArray, dNPrevArray, dNArray, dnNextArray, dTPrevArray, dtArray, dTNextArray)) #, dataMeanArray, dataStdevArray))

    # with open('dataExtractionCheck.csv', 'w', encoding='UTF8', newline='') as f:
    #     writer = csv.writer(f)

    #     # write the header
    #     writer.writerow(['targetNArray', 'targetTbArray', 'dNPrevArray', 'dNArray', 'dnNextArray', 'dTPrevArray', 'dtArray', 'dTNextArray'])

    #     # write multiple rows
    #     writer.writerows(checking)

    return targetNArray, targetTbArray, dNPrevArray, dNArray, dnNextArray, dTPrevArray, dtArray, dTNextArray, dataMeanArray, dataStdevArray

def computeDataMeanAndStdev(targetNArray, targetTbArray, dNPrevArray, dNArray, dnNextArray, dTPrevArray, dtArray, dTNextArray):
    dataMeanArray = []
    dataStdevArray = []

    # Calculate mean and standard deviation for targetNArray
    mean = statistics.mean(targetNArray)
    stdev = statistics.pstdev(targetNArray)
    dataMeanArray.append(mean)
    dataStdevArray.append(stdev)

    # Calculate mean and standard deviation for targetTbArray
    mean = statistics.mean(targetTbArray)
    stdev = statistics.pstdev(targetTbArray)
    dataMeanArray.append(mean)
    dataStdevArray.append(stdev)

    # Value from running the simulation on the same data set
    dataMeanArray.append(7.632571081)
    dataStdevArray.append(8.772972)

    # Calculate mean and standard deviation for dNPrevArray
    mean = statistics.mean(dNPrevArray)
    stdev = statistics.pstdev(dNPrevArray)
    dataMeanArray.append(mean)
    dataStdevArray.append(stdev)

    # Calculate mean and standard deviation for dNArray
    mean = statistics.mean(dNArray)
    stdev = statistics.pstdev(dNArray)
    dataMeanArray.append(mean)
    dataStdevArray.append(stdev)

    # Calculate mean and standard deviation for dnNextArray
    mean = statistics.mean(dnNextArray)
    stdev = statistics.pstdev(dnNextArray)
    dataMeanArray.append(mean)
    dataStdevArray.append(stdev)

    # Calculate mean and standard deviation for dTPrevArray
    mean = statistics.mean(dTPrevArray)
    stdev = statistics.pstdev(dTPrevArray)
    dataMeanArray.append(mean)
    dataStdevArray.append(stdev)

    # Calculate mean and standard deviation for dtArray
    mean = statistics.mean(dtArray)
    stdev = statistics.pstdev(dtArray)
    dataMeanArray.append(mean)
    dataStdevArray.append(stdev)

    # Calculate mean and standard deviation for dTNextArray
    mean = statistics.mean(dTNextArray)
    stdev = statistics.pstdev(dTNextArray)
    dataMeanArray.append(mean)
    dataStdevArray.append(stdev)

    # Value from running the simulation on the same data set
    dataMeanArray.append(-6.155390779)
    dataStdevArray.append(68.41398)

    # print("dataMeanLenght", len(dataMeanArray))
    # print(dataMeanArray)
    # print("dataStdevArray length", len(dataStdevArray))
    # print(dataStdevArray)
    return dataMeanArray, dataStdevArray

def normalize(data, mean, stdev):
    return (data-mean) / stdev

def denormalise_data(engine_prediction):  
    # Step 1 - LSTM Normalisation Constants
    # ------------

    # Note: STD of engine speed training data
    std_training = 399.89592366
    # Note: Mean of engine speed training data 
    mean_training = 1344.84139439

    denormalised_engine_speed_prediction = engine_prediction*std_training+mean_training
    return denormalised_engine_speed_prediction

def PIDSimulation(PIDParameters, dataFrameInput):
    # Step 1 - PID Parameters/Coefficients
    # ------------
    print("abcd")
    kP = PIDParameters[0];
    kI= PIDParameters[1];
    kD = PIDParameters[2];

    # ------------
    # Step 1 - END

    # -------------------------------------------
    # CONTROL CONSTANTS - START
    # -------------------------------------------
    # DEFINING ROWS AND COLUMNS FOR BILINEAR INTERPOLATION
    engine_speed_columns=[750, 875, 1000, 1125, 1250, 1375, 1500, 1625, 1750, 1875, 2000, 2125, 2250, 2375, 2500, 2635, 2750, 2875, 3000, 3125]
    torque_rows=[4.9, 50, 100, 150, 200, 250, 300]
    idling_N = 750
    # -------------------------------------------
    # CONTROL CONSTANTS - END
    # -------------------------------------------

    # -------------------------------------------
    # LSTM CONSTANTS & MODEL - START
    # -------------------------------------------

    # Load Saved Model
    savedModel = models.load_model('pid_emulate.h5', custom_objects=None, compile=True, options=None)

    # Initialising constants and empty arrays to plot output
    pedal_array = []
    predicted_engine_speed_array = []
    predicted_engine_speed_array_output = []
    target_N_array = []
    kI_action_array = []
    kP_action_array = []
    kD_action_array = []
    total_pedal_position = 0 
    error_previous_array = []
    targetNArrayOutput = []

    sequence_length = 1
    batch_size = 20000
    sampling_rate = 1

    time_iteration = len(dataFrameInput)
    # print('Length', time_iteration)
    # ------------
    # Step 1 - END

    # ------------
    # Input PARAMETERS - END


    # LSTM Dataframe - START
    # ------------

    lstm_df = pd.DataFrame(columns = ['Target_N', 'Target_Tb', 'pedal', 'dN_prev', 'dN', 'dN_Next', 'dT_Prev', 'dT', 'dT_Next', 'Error_Prev' ])
    # ------------
    # LSTM Dataframe  - END

    # target engine speed, torque, pedal, dT, dT Next
    # mean_all_data = [1.34754509e+03, 5.72366989e+01, 1.46944703e+01, 2.35267857e-02, -1.27232143e-02, 1.84375000e-02, 1.28343051e-02, 6.90854699e-03, 7.47315599e-03, 2.14469619e+01]

    # std_all_data = [389.29599207, 52.62897013, 12.41536045, 128.88844119, 128.80488408, 191.02436595, 27.53056784, 16.32363125, 16.32370291, 204.67945279]

    targetNArray, targetTbArray, dNPrevArray, dNArray, dnNextArray, dTPrevArray, dtArray, dTNextArray, dataMeanArray, dataStdevArray = extractFromDataFrameInput(dataFrameInput)

    mean_all_data = [1.34754509e+03, 5.72366989e+01, 1.46944703e+01, 2.35267857e-02, -1.27232143e-02, 1.84375000e-02, 1.28343051e-02, 6.90854699e-03, 7.47315599e-03, 2.14469619e+01]

    std_all_data = [389.29599207, 52.62897013, 12.41536045, 128.88844119, 128.80488408, 191.02436595, 27.53056784, 16.32363125, 16.32370291, 204.67945279]

    actual_N_previous = 800
    sum_error = 0
    for row in range(0, time_iteration):
        print(row)
        # DRIVE CYCLE TRACE INPUTS
        current_time_step_inputs = [targetTbArray[row],targetNArray[row]]
        if( row != time_iteration - 1):
            next_time_step_inputs = [targetTbArray[row + 1],targetNArray[row+1]]
        else:
            next_time_step_inputs = current_time_step_inputs
        # FEEDFORWARD # 1 - Baseline Pedal Position
        baseline_pedal_position = bilinear_interpolation(torque_rows, engine_speed_columns, current_time_step_inputs)

        # Note: Condition for enabling baseline pedal 
        if (abs(next_time_step_inputs[1]-current_time_step_inputs[1] > 1) or abs(next_time_step_inputs[0]-current_time_step_inputs[0] > 1)):
            baseline_pedal_position_help = 1
            baseline_pedal_position = baseline_pedal_position

        # FEEDFORWARD # 2 - Baseline Pedal Advance
        baseline_pedal_position_advance = bilinear_interpolation(torque_rows, engine_speed_columns, next_time_step_inputs)

        # Note: Condition for enabling baseline pedal advance
        feedForward_condition_help = 0

        if(row > 0 and row != time_iteration - 1):
            if (targetNArray[row] == idling_N):
                #Previously this targetNPrevious = 1250 always, so feedforward condition help was always 0
                if ((targetNArray[row] == idling_N and targetNArray[row + 1] != idling_N) and (targetNArray[row - 1] == idling_N)):
                    feedForward_condition_help = 1
            else:
                feedForward_condition_help = 0

            if (feedForward_condition_help == 1):
                baseline_pedal_position_advance = baseline_pedal_position_advance
            else:
                baseline_pedal_position_advance = 0

        # CALCULATED VALUES

        if (row <2):
            previous_error = 0
        else:
            previous_error = error_previous_array[row-1]

        error = (targetNArray[row] - actual_N_previous) # Correct - target setpoint minus output engine speed

        error_previous_array.append(error) # Correct - Appends previous error to array of errors

        if (sum_error > 10000): # Correct - nullifies integrator action when exceeding 10000 and -10000
            sum_error = 10000
        elif (sum_error < -10000):
            sum_error = -10000

        dt = 0.1

        #Functional 1hz
        #kI_action = kI*(error/dt + sum_error)
        kI_action = kI*(error + sum_error)

        #Functional 1hz
        #sum_error = sum_error + error/dt
        sum_error = sum_error + error/dt

        #Functional 1hz
        #kD_action = kD*(error-previous_error)*dt
        kD_action = kD*(error-previous_error)*dt #correct
        #kD_action = 0
        kP_action = (error)*kP # correct

        if row == 0:
            kI_action = 0 
            kD_action = 0
        
        kI_action_array.append(kI_action)
        kD_action_array.append(kD_action)
        kP_action_array.append(kP_action)
        total_action_parallel = kP_action + kI_action + kD_action
        total_action_standard = ((kI_action + kD_action) + error)*kP
        #print('!!!!!!!! TIME !!!!!!!!!', row)
        #print('N- Target Previous', target_N)
        #print('N - Actual Previous', actual_N_previous)
        #print('PID - Total', total_action_parallel)
        #print('PID - D', kD_action)
        #print('PID - I', kI_action)
        #print('Sum error previous', sum_error_previous)
        #print('PID - P', kP_action)
        #print('PID - SUM ERROR', sum_error)
        #print('N - Error', error)
        #print('PEDAL - Baseline', baseline_pedal_position)
        if (feedForward_condition_help == 0):
            total_pedal_position = total_action_parallel + baseline_pedal_position
            #print('PEDAL - Triggered', total_pedal_position)
        else:
            #print('RICK ROSS')
            total_pedal_position = baseline_pedal_position_advance

        total_pedal_position = max(1.6, total_pedal_position)
        #print('Total PEDAL POSITION', total_pedal_position)
        
        #print('Pedal Position', total_pedal_position)
        #print('Actual N Previous', actual_N_previous)
        #print('error', error)
        #derivative_N_prev = prev_target_N - prev_two_target_N
        #derivative_N = target_N - prev_target_N
        #derivative_N_next = nxt_target_N - prev_target_N
        #derivative_T_prev = prev_target_Tb - prev_two_target_Tb
        #derivative_T = target_Tb - prev_target_Tb
        #derivative_T_next = nxt_target_Tb - prev_target_Tb
        #error 

        #if total_pedal_position>10:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
        #total_pedal_position = total_pedal_position*0.5
        # ---------------
        # LSTM Actual Engine Speed Prediction Module - Start
        # ---------------

        #input_prediction = [target_N, target_Tb, total_pedal_position, derivative_N_prev, derivative_N, derivative_N_next, derivative_T_prev, derivative_T, derivative_T_next]
    
        if row == 0:
            #print('Triggered')
            new_row_df = pd.DataFrame([{
            'Target_N': targetNArray[row], 
            'Target_Tb': targetTbArray[row], 
            'pedal': total_pedal_position, 
            'dN_prev': 0, 
            'dN': 0, 
            'dN_Next': 0, 
            'dT_Prev': 0, 
            'dT': 0, 
            'dT_Next': 0, 
            'Error_Prev': 0
        }])
            lstm_df = pd.concat([lstm_df, new_row_df], ignore_index = True)
            # lstm_df = lstm_df.append({'Target_N': targetNArray[row], 'Target_Tb': target_Tb, 'pedal': total_pedal_position, 'dN_prev': 0, 'dN': 0, 'dN_Next': 0, 'dT_Prev': 0, 'dT': 0, 'dT_Next': 0, 'Error_Prev': 0}, ignore_index = True)
            lstm_df = pd.concat([lstm_df, new_row_df], ignore_index = True)
            # lstm_df = lstm_df.append({'Target_N': target_N, 'Target_Tb': target_Tb, 'pedal': total_pedal_position, 'dN_prev': 0, 'dN': 0, 'dN_Next': 0, 'dT_Prev': 0, 'dT': 0, 'dT_Next': 0, 'Error_Prev': 0}, ignore_index = True)
            #print('Triggered DF', lstm_df)
            lstm_df.head()
        else:
            new_row_df = pd.DataFrame([{
            'Target_N': targetNArray[row],
            'Target_Tb': targetTbArray[row],
            'pedal': total_pedal_position,
            'dN_prev': dNPrevArray[row],
            'dN': dNArray[row],
            'dN_Next': dnNextArray[row],
            'dT_Prev': dTPrevArray[row],
            'dT': dtArray[row],
            'dT_Next': dTNextArray[row],
            'Error_Prev': -error
            }])
            lstm_df = pd.concat([lstm_df, new_row_df], ignore_index = True) 
            # lstm_df = lstm_df.append({'Target_N': target_N, 'Target_Tb': target_Tb, 'pedal': total_pedal_position, 'dN_prev': derivative_N_prev, 'dN': derivative_N, 'dN_Next': derivative_N_next, 'dT_Prev': derivative_T_prev, 'dT': derivative_T, 'dT_Next': derivative_T_next, 'Error_Prev': -error}, ignore_index = True)
            #print('INPUT RAW', lstm_df)
            if len(lstm_df.index) == 99:
                lstm_df = lstm_df.iloc[96:, :]
        #print('LSTM DF', lstm_df)
        lstm_input = lstm_df[['Target_N','Target_Tb','pedal', 'dN_prev', 'dN', 'dN_Next', 'dT_Prev', 'dT', 'dT_Next', 'Error_Prev']]
        #print('LSTM Input', lstm_input)
        # LSTM Actual Engine Speed prediction
        lstm_normalized_input = normalize(lstm_input.values, mean_all_data, std_all_data)

        df_lstm_normalized_input = pd.DataFrame(lstm_normalized_input)
        #print('Input', df_lstm_normalized_input)
        x_input = df_lstm_normalized_input[[1,2,3,4,5,6,7,8,9,0]].values
        y_input = np.random.randint(10,90,(100))
        
        x_input = np.array(x_input)
        y_input = np.array(y_input)
        x_input = x_input.astype(np.float32)
        y_input = y_input.astype(np.float32)
        dataset_test = preprocessing.timeseries_dataset_from_array(
                    x_input,
                    y_input,
                    sampling_rate=1,
                    sequence_stride=1,
                    batch_size=batch_size,
                    sequence_length=1
        )

        #prediction = savedModel.predict(normalized_input)
        
        x = []
        x_trial = []
        y_trial = []
        y = []
        count = 0 
        #print("HOLLA", dataset_test)
        for x_tf, y_tf in dataset_test:

            #print(len(dataset_test))
            count = count + 1 
            #print(count)
            for package in x_tf:
                x_trial.append(package)
                #print("bitch",package)
            
            for package_y in y_tf:
                y_trial.append(package_y)

            #if count == 16:
                #x = x_tf
                #y = y_tf
                #print('x_tf', x_tf)

        x_trial = np.array(x_trial)
        y_trial = np.array(y_trial)
        model_prediction = savedModel.predict(x_trial)

        #plt.close()
        predicted_engine_speed = denormalise_data(model_prediction) 
        actual_N_previous = predicted_engine_speed[-1][0]
        N_previous = predicted_engine_speed
        #print(type(actual_N_previous))
        predicted_engine_speed_array.append(actual_N_previous)
        pedal_array.append(total_pedal_position)
        targetNArrayOutput.append(targetNArray[row])
        #print('y_pred_array', N_previous)
        #print('pedal array', pedal_array)
        #plt.figure(figsize=(8, 6), dpi=80)
        #plt.plot(predicted_engine_speed_array_output,'k-',label='$Prediction$ (rpm)')
        #plt.plot(target_N_array,'r-',label='$target$ (rpm)')
        #figManager = plt.get_current_fig_manager()
        #figManager.window.wm_geometry("-1000+0")
    
        #fig.canvas.start_event_loop(0.001) 
        #plt.pause(0.001)

        #if row_int == 0:
        #    plt.show()
        

        # STEP 1
        # Data parsing and formatting for TensorFlow  

        # ---------------
        # LSTM Actual Engine Speed Prediction Module - End
        # ---------------

    y_output = np.column_stack((predicted_engine_speed_array, targetNArrayOutput, pedal_array, error_previous_array))

    with open('Control_Output_LSTM.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(['LSTM Engine Speed', 'Target Engine Speed', 'Pedal Output', 'error_previous_array'])

        # write multiple rows
        writer.writerows(y_output)

    x_putput = np.column_stack((kP_action_array, kI_action_array, kD_action_array,))

    with open('PID_action_output.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(['P Action', 'I Action', 'D Action'])

        # write multiple rows
        writer.writerows(x_putput)

    print('y_pred_array', len(predicted_engine_speed_array))
    print('pedal array', len(pedal_array))
    return predicted_engine_speed_array
