import schedule
import time



def message():
#=====================================Ai Code====================================
    from pandas import read_csv
    import datetime
    from pandas import concat
    import numpy as np
    # load data
    def parse(x):
        return datetime.datetime.strptime(x, '%Y %m %d %H')

    # 첫 주행 날짜

    first_date         = datetime.datetime.strptime("20221101", "%Y%m%d")
    first_date_str     = str(first_date)
    first_date_str     = first_date_str[:first_date_str.find(' ')].replace('-', '')
    print(first_date_str)
    print(first_date)

    # 현재 날짜
    # now_date           = datetime.datetime.now()

    # # 받아올 데이터 일수
    # read_time          = str(now_date - first_date)
    # print(read_time)
    # read_time          = int(read_time[:read_time.find(' ')])
    # print(read_time)

    dataset = []

    # 역대 환경 데이터를 모두 읽어옴
    for i in range(28):		#(read_time)
        read_file_name = str(first_date + datetime.timedelta(days = i))
        read_file_name = read_file_name[:read_file_name.find(' ')].replace('-', '')
        print(read_file_name)
        dataset.append(read_csv('H:\\My Drive\\dataset' + '\\fake_data_' + read_file_name +'.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse))

    dataset            = concat([dataset[i] for i in range(len(dataset))], axis=0)
    dataset.drop('No', axis=1, inplace=True) 
    # manually specify column names
    dataset.columns = ['local', 'temp', 'humidity', 'metter']
    dataset.index.name = 'date'

    # print dataset
    print("================================ datased head 5 ================================")
    print(dataset.head(5)) 
    # save to file
    dataset.to_csv('H:\\My Drive\\dataset\\pollution.csv')
    print("================================================================================")
    print("\n\n\n")



    # prepare data for lstm
    from pandas import DataFrame
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import MinMaxScaler

    # convert series to supervised learning
    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        # print(df)
        cols, names = list(), list()
        # 7일 전의 환경 데이터를 input, 7일 후의 환경 데이터를 target으로 설정
        cols.append(df.shift(6*13*7))
        names += [('var%d(t-%d)' % (j+1, 7)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        cols.append(df)
        names += [('var%d(t)' % (j+1)) for j in range(n_vars)]

        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        # drop columns we don't want to predict
        agg.drop(agg.columns[[3,4]], axis=1, inplace=True)
        
        # 구역에 따라  환경데이터를 묶어준다.
        local = agg[['var1(t-7)',  'var2(t-7)',  'var3(t-7)']].values
        local = local.reshape(13*int((local.shape[0]/13/6)),6,3)
        label = agg['var3(t)'].values
        label = label.reshape(13*int((label.shape[0]/13/6)),6)
        return local, label

    # load dataset
    dataset = read_csv('H:\\My Drive\\dataset\\pollution.csv', header=0, index_col=0)
    values = dataset.drop(['local'], axis = 1).values
    # integer encode direction
    encoder = LabelEncoder()
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed, target = series_to_supervised(scaled, 1, 1)


    print("======================== reframed.shape, target.shape ==========================")
    print(reframed.shape, target.shape)
    print("================================================================================")



    # split into train and test sets
    values = reframed
    # 5일치를 학습시킴
    n_train_hours = int(values.shape[0]/13/3)*2*13
    train_X, train_y = values[:n_train_hours], target[:n_train_hours]
    test_X,  test_y  = values[n_train_hours:], target[n_train_hours:]

    print("============================== Processing Data =================================")
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    print(train_X, train_y)
    print("================================================================================")


    from keras.models import Sequential
    from keras.layers import Dense, LSTM
    from keras.layers import BatchNormalization
    import matplotlib.pyplot as plt
    # design network
    model = Sequential()
    model.add(LSTM(25, input_shape=(train_X.shape[1], train_X.shape[2]), activation="softsign", recurrent_activation="elu"))
    model.add(Dense(12))
    model.add(Dense(3))
    model.add(Dense(8))
    model.add(Dense(6))
    model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
    # fit network
    history = model.fit(train_X, train_y, epochs=200 , batch_size=6, validation_data=(test_X, test_y), verbose=2, shuffle=False, validation_split = 0.1)
    # plt history
    # plt.xlabel("epoch")
    # plt.ylabel("loss")
    # plt.plot(history.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='test')
    # plt.legend()
    # plt.show()



    from sklearn.metrics import mean_squared_error
    # # make a prediction
    # yhat = model.predict(test_X)
    # test_X = test_X.reshape((test_X.shape[0]*test_X.shape[1], test_X.shape[2]))
    # yhat = yhat.reshape(yhat.shape[0]*yhat.shape[1],1)
    # print(test_X.shape, yhat.shape)

    # # invert scaling for forecast
    # inv_yhat = np.concatenate(( test_X[:,:-1],yhat), axis = 1)
    # print(inv_yhat.shape)
    # inv_yhat = scaler.inverse_transform(inv_yhat)
    # inv_yhat = inv_yhat[:,-1]
    # # invert scaling for actual

    # test_y = test_y.reshape(test_y.shape[0]*test_y.shape[1], 1)
    # inv_y = np.concatenate((test_X[:,:-1], test_y), axis=1)
    # inv_y = scaler.inverse_transform(inv_y)
    # inv_y = inv_y[:,-1]

    prediction = model.predict(test_X)

    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(test_y, prediction))
    print('Test RMSE: %.3f' % rmse)

    loss, acc = model.evaluate(test_X, test_y, batch_size=1)
    print("loss : ", loss)
    print("acc : " , acc)

    # R2 구하기
    from sklearn.metrics import r2_score
    r2_y_predict = r2_score(test_y, prediction)
    print("R2 : ", r2_y_predict)

    print("============================== Processing Data =================================")
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    print(train_X, train_y)
    print("================================================================================")



    # 다음 날의 주행을 예측하기 위해서는 현재로부터 6일 전의 데이터를 넣고 추출시켜야 함
    tst_tim = str(datetime.datetime.now() - datetime.timedelta(days = 6))
    tst_tim = tst_tim[:tst_tim.find(' ')].replace('-', '')
    print(tst_tim)
    predic_data = []
    predic_data.append(read_csv('H:\\My Drive\\dataset' + '\\fake_data_' + "20221128" +'.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse))


    predic_data = concat([predic_data[i] for i in range(len(predic_data))], axis=0)
    predic_data.drop('No', axis=1, inplace=True) 
    # manually specify column names
    predic_data.columns = ['local', 'temp', 'humidity', 'metter']
    predic_data.index.name = 'date'

    print(predic_data.head(5)) 

    predic_values = predic_data.drop(['local'], axis = 1).values
    # integer encode direction
    encoder = LabelEncoder()
    # ensure all data is float
    predic_values = predic_values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    predic_scaled = scaler.fit_transform(predic_values)
    # frame as supervised learning
    predic_scaled = predic_scaled.reshape(13,6,3)
    predic = np.array(model.predict(predic_scaled))
    print(predic_scaled)

    #주행 스케쥴
    #softmax function 
    def softmax(a):
        exp_a = np.exp(a)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y

    # sults clean_time
    clean_time = np.array([])
    local = np.array([])
    for i in predic:
        clean_time = np.append(clean_time, [int(round(i)) for i in (softmax(i)*60)])
        local = np.append(local, np.arange(1,7))
    clean_time = clean_time.reshape(clean_time.shape[0],1)
    local      = local.reshape(local.shape[0],1)
    scadul     = np.concatenate((local, clean_time), axis = 1)
    columns    = [ 'local', 'stay_time']
    send_df    = DataFrame(scadul, columns = columns)
    send_df    = send_df.set_index('local')
    send_df.to_csv('H:\\My Drive\\dataset\\send.csv')

    # compare prediction and db 
    tomorrow_time = str(datetime.datetime.now() + datetime.timedelta(days = 1))
    tomorrow_time = tomorrow_time[:tomorrow_time.find(' ')].replace('-', '')
    tomorrow_data=np.array([])
    tomorrow_data = np.append(tomorrow_data,read_csv('H:\\My Drive\\dataset' + '\\fake_data_' + "20221128" +'.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)['metter'].values)

    real_clean_time = np.array([])
    for i in predic:
        real_clean_time = np.append(real_clean_time, [int(round(i)) for i in (softmax(i)*60)])
    
    real_clean_time = real_clean_time.reshape(real_clean_time.shape[0],1)
    rmse = np.sqrt(mean_squared_error(real_clean_time, clean_time))

    #예측 결과 100% 맞음
    #다만, 현재 예측한 결과는 train_data에 이미 있는 값임
    time = np.concatenate((real_clean_time, clean_time), axis = 1)
    print(time)
    print('Test RMSE: %.3f' % rmse)
#=====================================Ai Code====================================



#=====================================Web Code===================================
    # [0] 모듈 import 
    import pandas as pd
    import numpy as np
    import datetime


    # [1] t-7일 날짜 생성
    day_1      = datetime.datetime.now()-datetime.timedelta(weeks=1)
    day_2      = day_1.strftime("%Y%m%d")


    # [2] 실행 결과로 파일에 저장될 값들 저장할 배열 생성
    count    = 0
    temp     = []
    humidity = []
    metter   = []

    # [3] 일주일 단위로 DashBoard에서 사용할 거임 -> (count < 7)
    while(count < 7):
        if (count == 0):
            file = "H:\\My Drive\\GreenAI_dataset\\fake_data_{}.csv".format(day_2)
            df = pd.read_csv(file)

            print(day_2)
            print(df)

            # 온도 value 추출
            avg_temp = df.temp  
            avg_temp = avg_temp.tolist()
            avg_temp = round(np.mean(avg_temp),1)
            temp.append(avg_temp)


            # 습도 value 추출
            avg_humidity = df.humidity  
            avg_humidity = avg_humidity.tolist()
            avg_humidity = round(np.mean(avg_humidity),1)
            humidity.append(avg_humidity)


            # 미세먼지 value 추출
            avg_metter = df.metter  
            avg_metter = avg_metter.tolist()
            avg_metter = round(np.mean(avg_metter),1)
            metter.append(avg_metter)


            day_c1 = day_1
            count = count + 1

        else:
            day_c1 = day_c1 + datetime.timedelta(days=1)
            day_c2 = day_c1.strftime("%Y%m%d")

            file = "H:\\My Drive\\GreenAI_dataset\\fake_data_{}.csv".format(day_c2)
            df = pd.read_csv(file)

            print(day_c2)
            print(df)

            # 온도 value 추출
            avg_temp = df.temp  
            avg_temp = avg_temp.tolist()
            avg_temp = round(np.mean(avg_temp),1)
            temp.append(avg_temp)


            # 습도 value 추출
            avg_humidity = df.humidity  
            avg_humidity = avg_humidity.tolist()
            avg_humidity = round(np.mean(avg_humidity),1)
            humidity.append(avg_humidity)


            # 미세먼지 value 추출
            avg_metter = df.metter  
            avg_metter = avg_metter.tolist()
            avg_metter = round(np.mean(avg_metter),1)
            metter.append(avg_metter)
            count = count + 1

    print(temp)
    print(humidity)
    print(metter)


    # [4] 숫자(Sensor Data Avg) -> 문자열 변환 
    temp = ','.join(map(str, temp))
    humidity = ','.join(map(str, humidity))
    metter = ','.join(map(str, metter))


    # [5] 파일 생성 후 결과 저장 
    f = open("C:\\Users\\ewqds\\Documents\\GitHub\\D_web\\index.js", "w", encoding="utf-8")
    f.write("function temp_data() {" + "\n\t"+ "return " + "[" + temp + "]" + "\n" + "}" + "\n\n\n")
    f.close()

    f = open("C:\\Users\\ewqds\\Documents\\GitHub\\D_web\\index.js", "a", encoding="utf-8")
    f.write("function humidity_data() {" + "\n\t"+ "return " + "[" + humidity + "]" + "\n" + "}" + "\n\n\n")
    f.close()

    f = open("C:\\Users\\ewqds\\Documents\\GitHub\\D_web\\index.js", "a", encoding="utf-8")
    f.write("function metter_data() {" + "\n\t"+ "return " + "[" + metter + "]" + "\n" + "}" + "\n\n")
    f.close()



schedule.every().day.at("04:40").do(message)

while True:
    schedule.run_pending()
    time.sleep(1)