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
        file = "G:\\내 드라이브\\web\\dataset\\fake_data_{}.csv".format(day_2)
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

        file = "G:\\내 드라이브\\web\\dataset\\fake_data_{}.csv".format(day_c2)
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
f = open("index.js", "w", encoding="utf-8")
f.write("function temp_data() {" + "\n\t"+ "return " + "[" + temp + "]" + "\n" + "}" + "\n\n\n")
f.close()

f = open("index.js", "a", encoding="utf-8")
f.write("function humidity_data() {" + "\n\t"+ "return " + "[" + humidity + "]" + "\n" + "}" + "\n\n\n")
f.close()

f = open("index.js", "a", encoding="utf-8")
f.write("function metter_data() {" + "\n\t"+ "return " + "[" + metter + "]" + "\n" + "}" + "\n\n")
f.close()