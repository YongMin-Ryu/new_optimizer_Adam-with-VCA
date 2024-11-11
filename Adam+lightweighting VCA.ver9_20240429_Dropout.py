import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import csv
import math

def pt(x): #텐서 내 값 출력
    with tf.compat.v1.Session() as sess:
        return sess.run(tf.constant(x))

#<<<<<<< 학습 및 예측 데이터 구축 >>>>>>
input = pd.read_csv('Input_learn_day_Nomalization_Case13.csv')
output = pd.read_csv('Output_learn_day_Nomalization.csv')

input = np.array(input)
output = np.array(output)
test_input = pd.read_csv('Input_test_day_Nomalization_Case13.csv')
test_output = pd.read_csv('Output_test_day_Nomalization.csv')

Drop_Rate=0.1
Case=13

print(input.shape)
print(output.shape)

test_input = np.array(test_input)
 
xx = input.shape[1]

input = input.astype(float)

output = output.astype(float)
test_input = test_input.astype(float)
test_output = test_output.astype(float)

#<<<<<<<< 새로운 해를 모델에 적용하는 함수 정의 >>>>>>>>>>
def weight_update(W_new):
  count = 0
  for i in range(n_layer):
    for k in range(shape[i]):
      for l in range(shape[i+1]):
        W0[i*2][k][l] = W_new[count]  # W0의 짝수에 가중치 저장
        count += 1
    for n in range(shape[i+1]):
      W0[i*2+1][n] = W_new[count]     # W0의 홀수에 편향 저장
      count += 1
  return model.set_weights(W0)

#<<<<<<<<<< 기존 optimizer를 적용하는 함수 정의 >>>>>>>>>>>>>
def opti(x, y):
  with tf.GradientTape() as tape:
    pred = tf.cast(model(x), tf.float32)
    loss = tf.cast(loss_function(y, pred), tf.float32)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_vaiables))


#<<<<<<<<<< 목적함수에 대한 오차를 계산하는 함수 정의>>>>>>>>>>>>>
def train_loss():
    pred = model(input) # 1. 예측 (prediction)
    loss = loss_function(output, pred) # 2. Loss 계산
    loss = tf.cast(loss, tf.float32)
    #loss = 1.0000
    return loss
    #return pt(loss)

#<<<<<<<<<< 기존 optimizer를 적용하여 생성된 가중치를 복사하는 함수 정의>>>>>>>>>>>>>
def weight_copy():
  w_old = model.get_weights()
  w_new = np.zeros(n_weight, dtype = np.float32)
  count = 0
  for i in range(n_layer):
    for k in range(shape[i]):
      for l in range(shape[i+1]):
        w_new[count] = w_old[2*i][k][l]
        count += 1
    for n in range(shape[i+1]):
      w_new[count] = w_old[2*i+1][n]
      count += 1
  return w_new

for SA in range(0, 5):
    time = 10
    #<<<<<<< 신경망 매개변수 설정 >>>>>>
    g_loss=[]
    g_pred=[]
    g_geum=[]
    loss_function = tf.keras.losses.MeanSquaredError()      # 목적함수 설정
    af = 'relu'                                             # 활성화 함수 설정
    hidden_layer = 5                                       # 은닉층 개수 설정
    #hidden = [15, 15, 15, 15, 15]                                           # 각 은닉층의 노드 개수 설정
    hidden = [10, 10, 10, 10,10]
    epoch = 1200
    n_layer = hidden_layer + 1                              #
    shape = [xx] + hidden[:5] + [1]                             # 각 층의 노드를 shape에 저장
    n_weight = 0                                            
    for i in range(n_layer):                                # n_weight : 가중치와 편향의 전체 개수
      n_weight += (shape[i]+1) * shape[i+1]
    
    #<<<<<<< 신경망 모델 구축 >>>>>>
    for t in range(0, time):
      a_loss = []
      X = tf.keras.layers.Input(shape = [xx])
      if hidden_layer == 0:
        Y = tf.keras.layers.Dense(1)(X)
    
      elif hidden_layer == 1:
        H = tf.keras.layers.Dense(hidden[0], activation = af)(X)
        Y = tf.keras.layers.Dense(1)(H)
    
      else:
        H = tf.keras.layers.Dense(hidden[0], activation = af)(X)
        H = tf.keras.layers.Dropout(Drop_Rate)(H)
        for i in range(hidden_layer-1):
          H = tf.keras.layers.Dense(hidden[i+1], activation = af)(H)
          H = tf.keras.layers.Dropout(Drop_Rate)(H)
        Y = tf.keras.layers.Dense(1)(H)
    
      model = tf.keras.models.Model(X, Y)
      model.compile(loss = 'mse', optimizer = 'Adam')
      model.summary()
      
      W0 = model.get_weights()   #초기 가중치 및 편향을 W0 에 저장
     
    
      #<<<<<<<HS 매개변수 설정>>>>>>
      HMS = 4
      #HMCR= HMCR_list[SA]
      AR_max = 0.9
      AR_min = 0.9
      PAR_max = 0.3
      PAR_min = 0.05
      Bw_min = 0.00001
      Bw_max = 0.001
      c = math.log(Bw_min / Bw_max) / epoch
      ub = 1
      lb = -1
  #<<<<<<<VCA 매개변수 설정>>>>>>
      #idr1=0.5
      #idr2=0.5
      ar=0.1
      br=0.01
  #af=45
      cf=40 #cf는 원래 40

  #초기 Division Rate 설정 시작============VCA
      idr1=0.0
      #idr1=0.6
      idr2=0.9  #dr2는 원래 0.9
      
      #idr2=SA*0.1
  #for i in range(0, HMS):
  
      #for j in range(0,HMS):
  #  dr1.append(idr1)
  #  dr2.append(idr2)
  #초기 Division Rate 설정 끝 ============VCA

  # ch =============================VCA


  # MTF계산용 행렬 =============================VCA
      dist=[]
      dx=[]
      ang=[]    
      #<<<<<<<<< 초기 HM 생성 >>>>>>>>>
      HM = []
      for i in range(0, HMS-1):
        HM.append([])
    
      W = weight_copy()
      W = W.tolist()
      
      HM.append(W)
      for i in range(0, HMS-1):
        for j in range(0, n_weight):
          HM[i].append(random.uniform(lb, ub))
    
      #<<<<<<<<< 초기 HM의 오차 계산 >>>>>>>>>
      for i in range(0, HMS):
        weight_update(HM[i])
        HM[i].append(train_loss())
      
      #<<<<<<<<< HM 정렬 >>>>>>>>>
      HM.sort(key = lambda x:x[n_weight])
      print("초기 MSE = ", HM[0][n_weight])
      a_loss.append(HM[0][n_weight])
    
      dr1=idr1
      dr2=idr2
    #<<<<<<<<< 반복학습 실시 >>>>>>>>>
      print(n_weight)
      for epo in range(epoch):
        fi=[]
        pfi=[]
        sel=[]
        sel_1=[]  

          
      #<<<<<<<<< dr rebounding >>>>>>>>>  
        if dr1<=0 or dr1>1:
          dr1=idr1  
        if dr2<=0 or dr1>2:
          dr2=idr2        
      #<<<<<<<<< 선택확률 누적 >>>>>>>>>  
        sumfi=0  
          
        for i in range(1,HMS+1):
            #fi.append(HMS/i)
            fi.append(HMS/i)
            fii=HMS/i    
            sumfi += fii
          
        for i in range(0,HMS):
            pfi.append(fi[i]/sumfi)
              
          
        sell=0
        for i in range(0,HMS): 
            sell=sell+pfi[i]
            sel.append(sell)  
        
        
        ch=0
        AR = AR_max - (AR_max-AR_min)/epoch * epo
        #AR=0
        if random.random() < AR:
          a = 1
          if epo == 0:
            weight_update(HM[0][:n_weight])
          else:
            weight_update(new_HM1[:n_weight])
    
          model.fit(input, output, epochs=1, verbose = 0)  
          new_HM1 = weight_copy()     
          new_HM1 = new_HM1.tolist()
        
        else:
          optimizer = tf.keras.optimizers.Adam         # optimizer 설정
          #optimizer = tf.keras.optimizers.SGD(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-07)         # optimizer 설정
          #optimizer = tf.keras.optimizers.legacy.SGD         # optimizer 설정
          #optimizer = tf.keras.optimizers.SGD         # optimizer 설정
          PAR = PAR_min+(PAR_max - PAR_min) / epoch * epo
          Bw = Bw_max * np.exp(c*epo)
          a = 2
          new_HM2 = []
          if random.random() > dr1:
            for j in range(n_weight):
              #new_HM2.append(random.uniform(lb, ub))
              #new_HM2.append(random.uniform(lb, ub))
              ran= random.random()
              #print(sel[1])
              #for jj in range(HMS):
              #    if jj==HMS:
              #        jj=HMS
              for jj in range(HMS):
                  if ran<=sel[jj]:
                      new_HM2.append(HM[int(ran)][j])  #23.01.20 수정
                      break
                  #break
                      
              #if ran <= sel[0]:
              #    new_HM2.append(HM[int(ran)][j])
              #elif sel[0] < ran and ran <= sel[1]:
              #            #elif sel[HMS] < ran and ran <= sel[HMS+1]:
              #                #elif sel[0] < ran:
              #    new_HM2.append(HM[int(ran)][j])
         

              ch=1
          else:
            for j in range(0, n_weight):
              loc = random.randint(0, HMS-1)
              #new_HM2.append(HM[loc][j])
              if dr2 < random.random():
                  
                  new_HM2.append(random.uniform(HM[0][j], ub))
                 
                  #new_HM2.append(HM[0][j]+random.uniform(ub, lb))
                  ch=2
              else:
                  #new_HM2.append(HM[0][j]+random.uniform(ub, lb))
                  new_HM2.append(random.uniform(lb, HM[0][j]))
                  ch=3 

          #print(len(new_HM2))
          for i in range(0, n_weight):
              #new_HM2[i]+=random.random()*((1-epo/epoch)**cf)
              #new_HM2[i]+=(new_HM2[i]*random.random()*((1-epo/epoch)**cf))
              #new_HM2[i]+=(random.random()*((1-epo/epoch)**cf))/1000
              new_HM2[i]+=new_HM2[i]*(random.random()*((1-epo/epoch)**cf))
              #print((new_HM2[i]*random.random()*((1-epo/epoch)**cf))/10000)
              #new_HM2[i]+= random.uniform(-Bw, Bw)

    
          model.compile(loss = 'mse', optimizer = 'Adam')
          weight_update(new_HM2[:n_weight])
          model.fit(input, output, epochs=1, verbose = 0)
    
        #<<<<<<<<< 새로운 해와 기존해의 비교 >>>>>>>>>    
        if a == 1:
            weight_update(new_HM1)
            new_HM1.append(train_loss())
            if new_HM1[n_weight] < HM[HMS-1][n_weight]:
                HM[HMS-1] = new_HM1
   
        elif a == 2:
            weight_update(new_HM2)
            new_HM2.append(train_loss())
            
            if new_HM2[n_weight] < HM[HMS-1][n_weight]:
                
                HM[HMS-1] = new_HM2

        
            if ch==1:
                dr1-=0.01
                
            elif ch==2:
                    dr1+=0.01
                    dr2-=0.01

            elif ch==3:

                dr1+=0.01
                dr2+=0.01
    
        
        HM.sort(key = lambda x:x[n_weight])
        
          
        a_loss.append(HM[0][n_weight])
    
        if epo % 10 == 0:
          print("epoch = ", epo, "/", epoch, "MSE = ", HM[0][n_weight])
    
      print("time = ", t)
      #print(HM[0])
      #<<<<<<<<< 최적의 해를 이용한 예측실시 >>>>>>>>>
      weight_update(HM[0][:n_weight])
      pred = model.predict(test_input)
      pred = np.reshape(pred, (len(pred)))      
      pred = pred.tolist()
      
      geum = model.predict(input)
      geum = np.reshape(geum, (len(geum)))      
      geum = geum.tolist()


      g_pred.append(pred)
      g_geum.append(geum)
      g_loss.append(a_loss)
      
    g_pred = np.array(g_pred)
    g_pred = np.transpose(g_pred)

    g_geum = np.array(g_geum)
    g_geum = np.transpose(g_geum)
    
    g_loss = np.array(g_loss)
    g_loss = np.transpose(g_loss)

      

 
    #<<<<<<<<< 예측결과 csv 저장 >>>>>>>>>
    with open('Case%d, %d, 노드 10 , 댐 유입량 예측_AdamVCA_Dropput %f.csv' %(Case,SA, Drop_Rate),'a', newline='') as f:	
      wt = csv.writer(f)						
      for value in g_pred:						
        wt.writerow(value)
  
    #<<<<<<<<< 검증결과 csv 저장 >>>>>>>>>
    with open('Case%d, %d, 노드 10, 댐 유입량 검증_AdamVCA_Dropput %f.csv' %(Case,SA, Drop_Rate), 'a', newline='') as f:	
      wt = csv.writer(f)						
      for value in g_geum:						
        wt.writerow(value)				
  
    #<<<<<<<<< loss 값 csv 저장 >>>>>>>>>
  

    with open('Case%d, %d, 노드 10, 댐 유입량 MSE_AdamVCA_Dropput %f.csv' %(Case,SA, Drop_Rate), 'a', newline='') as f:		
        wt = csv.writer(f)						
        for value in g_loss:					
          wt.writerow(value)
        wt.writerow('')
