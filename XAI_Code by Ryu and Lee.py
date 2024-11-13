import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import numpy as np

import csv

#학습데이터 가져오기
input = pd.read_csv('Input_learn_day_Normalization_2024ver.csv')
output = pd.read_csv('Output_learn_day_Normalization_2024ver.csv')

#학습데이터 3차원으로 전환
x = len(output)
input = input[:x]
input = input.values.reshape(x, 17)

#테스트데이터 가져오기
test_input = pd.read_csv('Input_test_day_Normalization_2024ver.csv')
#test_output = pd.read_csv('12_ca_k.csv')

#테스트데이터 3차원 전환
y = len(test_input)
#test_output = test_output.values.reshape(y,1)
#test_input = test_input[:y]
test_input = test_input.values.reshape(y,17)

#t_output = output[0]
E_epoch=10

for tt in range(0,1):
  for t in range(0, 1):
    t_output = np.array(output.iloc[:, [t]])


    input_layer = tf.keras.layers.InputLayer(input_shape=(17,))
    H1 = tf.keras.layers.Dense(units = 10, activation = 'relu')
    H2 = tf.keras.layers.Dense(units = 10, activation = 'relu')
    H3 = tf.keras.layers.Dense(units = 10, activation = 'relu')
    H4 = tf.keras.layers.Dense(units = 10, activation = 'relu')
    H5 = tf.keras.layers.Dense(units = 10, activation = 'relu')
    output_layer = tf.keras.layers.Dense(units = 1)

    model = tf.keras.Sequential([input_layer, H1, H2, H3, H4, H5, output_layer])


    model.compile(loss='mse', optimizer = 'adam')


    results = model.fit(input, t_output, epochs=E_epoch)

    print(model.summary())

    predict  = model.predict(test_input)



    with open('%d MLP 대청댐 유입량 예측 결과_single_Epoch=%d.csv' %(tt, E_epoch), 'a', newline='') as f:
      wt = csv.writer(f)
      for value in predict:
        wt.writerow(value)

    geum = model.predict(input)

    with open('%d MLP 대청댐 유입량 검증 결과_single_Epoch=%d.csv' %(tt, E_epoch), 'a', newline='') as f:
      wt = csv.writer(f)
      for value in geum:
        wt.writerow(value)

    x = results.history['loss']
    y = np.array(x)
    y = y.reshape(len(x), 1)

    with open('%d MLP 대청댐 유입량 MSE 결과_single_Epoch=%d.csv' %(tt, E_epoch), 'a', newline='') as f:
      wt = csv.writer(f)
      for values in y:
        wt.writerow(values)



    #from google.colab import files

    #files.download('%d MLP 대청댐 유입량 예측 결과_single_Epoch=%d.csv' %(tt, E_epoch))
    #files.download('%d MLP 대청댐 유입량 검증 결과_single_Epoch=%d.csv' %(tt, E_epoch))
    #files.download('%d MLP 대청댐 유입량 MSE 결과_single_Epoch=%d.csv' %(tt, E_epoch))


    # 각 layer의 노드 출력값 확인

    weights = model.get_weights()

    layer1_model = tf.keras.Model(inputs = model.input, outputs = model.layers[0].output)
    layer1_model_output = layer1_model.predict(input)

    layer2_model = tf.keras.Model(inputs = model.input, outputs = model.layers[1].output)
    layer2_model_output = layer2_model.predict(input)

    layer3_model = tf.keras.Model(inputs = model.input, outputs = model.layers[2].output)
    layer3_model_output = layer3_model.predict(input)

    layer4_model = tf.keras.Model(inputs = model.input, outputs = model.layers[3].output)
    layer4_model_output = layer4_model.predict(input)

    layer5_model = tf.keras.Model(inputs = model.input, outputs = model.layers[4].output)
    layer5_model_output = layer5_model.predict(input)

    layer6_model = tf.keras.Model(inputs = model.input, outputs = model.layers[5].output)
    layer6_model_output = layer6_model.predict(input)


    layer6_model_output_T = np.transpose(layer6_model_output)

    #print(len(np.array(layer6_model_output).T[0]))          #np.array(list이름).T[열번호]  :  리스트의 열 추출

    z_1 = np.zeros(shape=(10, len(input)))
    for i in range(0, 10):
      for j in range(0, 17):
        z_1[i] += input.T[j] * weights[0][j][i] + weights[1][i]

    z_2 = np.zeros(shape=(10, len(input)))
    for i in range(0, 10):
      for j in range(0, 10):
        z_2[i] += np.array(layer1_model_output).T[j] * weights[2][j][i] + weights[3][i]

    z_3 = np.zeros(shape=(10, len(input)))
    for i in range(0, 10):
      for j in range(0, 10):
        z_3[i] += np.array(layer2_model_output).T[j] * weights[4][j][i] + weights[5][i]

    z_4 = np.zeros(shape=(10, len(input)))
    for i in range(0, 10):
      for j in range(0, 10):
        z_4[i] += np.array(layer3_model_output).T[j] * weights[6][j][i] + weights[7][i]

    z_5 = np.zeros(shape=(10, len(input)))
    for i in range(0, 10):
      for j in range(0, 10):
        z_5[i] += np.array(layer4_model_output).T[j] * weights[8][j][i] + weights[9][i]

    z_6 = np.zeros(shape=(1, len(input)))
    for i in range(0, 1):
      for j in range(0, 10):
        z_6[i] += np.array(layer5_model_output).T[j] * weights[10][j][i] + weights[11][i]


    #s_z_1 = sum(z_1)
    #s_z_2 = sum(z_2)
    #s_z_3 = sum(z_3)
    #s_z_4 = sum(z_4)
    #s_z_5 = sum(z_5)
    #s_z_6 = sum(z_6)
    #s_input = sum(input)




    layer5_R = np.zeros(shape=(10, len(input)))


    for j in range(0, 10):

        layer5_R[j] += ((z_5[j] * weights[10][j][0] + 0.001 * np.sign(z_6[0]) / 10) / (z_6[0] + 0.001 * np.sign(z_6[0]))) * layer6_model_output_T[0]

    #====================================================================================================================================================================================================================================

    layer4_R = np.zeros(shape=(10, len(input)))


    for j in range(0, 10):
      for i in range(0, 10):

        layer4_R[j] += ((z_4[j] * weights[8][j][i] + 0.001 * np.sign(z_5[i]) / 10) / (z_5[i] + 0.001 * np.sign(z_5[i]))) * layer5_R[i]

    #====================================================================================================================================================================================================================================

    layer3_R = np.zeros(shape=(10, len(input)))


    for j in range(0, 10):
      for i in range(0, 10):

        layer3_R[j] += ((z_3[j] * weights[6][j][i] + 0.001 * np.sign(z_4[i]) / 10) / (z_4[i] + 0.001 * np.sign(z_4[i]))) * layer4_R[i]

    #====================================================================================================================================================================================================================================

    layer2_R = np.zeros(shape=(10, len(input)))

    for j in range(0, 10):
      for i in range(0, 10):

        layer2_R[j] += ((z_2[j] * weights[4][j][i] + 0.001 * np.sign(z_3[i]) / 10) / (z_3[i] + 0.001 * np.sign(z_3[i]))) * layer3_R[i]

    #====================================================================================================================================================================================================================================

    layer1_R = np.zeros(shape=(10, len(input)))

    for j in range(0, 10):
      for i in range(0, 10):

        layer1_R[j] += ((z_1[j] * weights[2][j][i] + 0.001 * np.sign(z_2[i]) / 10) / (z_2[i] + 0.001 * np.sign(z_2[i]))) * layer2_R[i]

    #====================================================================================================================================================================================================================================

    input_layer_R = np.zeros(shape=(17, len(input)))

    for j in range(0, 17):
      for i in range(0, 10):

        input_layer_R[j] += ((np.array(input).T[j] * weights[0][j][i] + 0.001 * np.sign(z_1[i]) / 30) / (z_1[i] + 0.001 * np.sign(z_1[i]))) * layer1_R[i]

    #====================================================================================================================================================================================================================================


    input_layer_R = np.array(input_layer_R)
    input_layer_R = np.transpose(input_layer_R)


  with open('%d input_layer_R 출력값_single_Epoch=%d.csv' %(tt, E_epoch), 'a', newline='') as f:
    wt = csv.writer(f)
    for values in input_layer_R:
      wt.writerow(values)



  #files.download('%d input_layer_R 출력값_single_Epoch=%d.csv' %(tt, E_epoch))

