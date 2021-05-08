import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression
import numpy as np
from fbprophet import Prophet
from dateutil.parser import parse
import pandas as pd

def extractWebRequest(fileName):
    with open(fileName, "rb") as f:
        line = f.readline()
        reqSum = 0
        while line:
            items = line.split()
            reqSum += int(items[-2])
            line = f.readline()
    return reqSum

def getRequestFromDate(Date):
    prefix = "./decompressed/pagecounts-"
    result = []
    for hour in range(24):
        suffix = "-" + "{:02d}".format(hour) + "0000"
        fileName = prefix + Date + suffix
        reqSum = extractWebRequest(fileName)
        print([hour,reqSum])
        result.append(reqSum)
    return [Date,result]

def getARMA(webRequest):
    length = len(webRequest)
    result = [webRequest[0], webRequest[1], webRequest[2]]
    for i in range(3, length):
        estimatedWebRequest = webRequest[i-1] * 0.8 + webRequest[i-2] * 0.15 + webRequest[i-3] * 0.05
        result.append(estimatedWebRequest)
    return result

def getLoss(Y, estimatedY):
    length = len(Y)
    sumation = 0
    for i in range(length):
        sumation += abs(Y[i] - estimatedY[i])
    result = sumation / length
    return result

def getLRM(webRequest):
    x_const = np.array([[1],[2],[3]])
    print(x_const.shape)
    result = [webRequest[0], webRequest[1], webRequest[2]]
    length = len(webRequest)
    for i in range(3, length):
        y_temp = np.array(webRequest[i-3:i])
        logModel = LinearRegression()
        logModel.fit(x_const, y_temp)
        next_x = np.array([[4]])
        predictedY = logModel.predict(next_x)
        result.append(predictedY[0])

    return result

def transformPandas(recordList):
    c = parse(recordList[0][0]).replace(hour=5)
    data = {}
    ds = []
    y = []
    for date in range(0, 6):
        for h in range(0, 24):
            temp = parse(recordList[date][0]).replace(hour=h)
            ds.append(temp)
            y.append(recordList[date][1][h])
    data["ds"] = ds
    data["y"] = y
    df = pd.DataFrame(data)

    return df

def prophetPredict(df):
    prophet = Prophet()
    prophet.fit(df)
    future = prophet.make_future_dataframe(periods=1 * 24, freq='H', include_history=True)
    print(future)
    forcast = prophet.predict(future)
    yhat = forcast[["yhat_lower"]]
    y_value = yhat.values.tolist()
    return y_value

def getARMA2D(recordList):
    length = len(recordList)
    result = [recordList[6][1][0], recordList[6][1][1], recordList[6][1][2]]
    for i in range(3, 24):
        estimatedWebRequest1 = recordList[6][1][i-1] * 0.4 + recordList[6][1][i-2] * 0.075 + recordList[6][1][i-3] * 0.025
        estimatedWebRequest2 = recordList[5][1][i-1] * 0.4 + recordList[4][1][i-2] * 0.075 + recordList[3][1][i-3] * 0.025
        result.append(estimatedWebRequest1 + estimatedWebRequest2)
    return result

def getMax(webRequest):
    length = len(webRequest)
    result = [webRequest[0], max(webRequest[0],webRequest[1]), max(webRequest[0],webRequest[1],webRequest[2])]
    for i in range(3, length):
        estimatedWebRequest = max(webRequest[i-3:i])
        result.append(estimatedWebRequest)
    return result

def getMin(webRequest):
    length = len(webRequest)
    result = [webRequest[0], min(webRequest[0],webRequest[1]), min(webRequest[0],webRequest[1],webRequest[2])]
    for i in range(3, length):
        estimatedWebRequest = min(webRequest[i-3:i])
        result.append(estimatedWebRequest)
    return result

def plotTheWhole(recordList):
    plt.plot(recordList[0][1])
    plt.plot(recordList[1][1])
    plt.plot(recordList[2][1])
    plt.plot(recordList[3][1])
    plt.plot(recordList[4][1])
    plt.plot(recordList[5][1])
    plt.plot(recordList[6][1])

if __name__ == "__main__":
    # record5 = getRequestFromDate("20150105")
    # record6 = getRequestFromDate("20150106")
    # record7 = getRequestFromDate("20150107")
    # recordList = [record5, record6, record7]
    # with open("dateRequest2.dat", "wb") as f:
    #    pickle.dump(recordList, f)
    #
    # with open("dateRequest.dat", "rb") as f:
    #     recordList = pickle.load(f)

    recordList = []

    with open("dateRequest.dat", "rb") as f:
        recordList = pickle.load(f)

    print(recordList)

    buff = []
    for i in range(0, 7):
        buff.extend(recordList[i][1])

    # print(buff)
    ARMAResult = getARMA(buff)
    loss = getLoss(buff,ARMAResult)
    print("loss of ARMA:")
    print(loss)

    LRMResult = getLRM(buff)
    loss = getLoss(buff, LRMResult)
    print("loss of LDM:")
    print(loss)
    #
    # df = transformPandas(recordList)
    # Presult = prophetPredict(df)
    # Presult = Presult[-25:-1]
    # temp = [i[0] for i in Presult]
    # loss = getLoss(recordList[6][1], temp)
    # print("loss of Presult:")
    # print(loss)

    # ARMA2DResult = getARMA2D(recordList)
    # loss = getLoss(recordList[6][1], ARMA2DResult)
    # print("loss of ARMA2DResult:")
    # print(loss)

    MaxResult = getMax(buff)
    loss = getLoss(buff,MaxResult)
    print("loss of MaxResult:")
    print(loss)
    #
    MinResult= getMin(buff)
    loss = getLoss(buff,MinResult)
    print("loss of MinResult:")
    print(loss)

    plt.plot(buff)
    plt.plot(ARMAResult)
    plt.plot(LRMResult)
    plt.plot(MaxResult)
    plt.plot(MinResult)
    #plt.plot(Presult)

    plt.legend(["origin", "ARMA", "LRMResult", "MaxResult","MinResult"])
    plt.xlabel("time")
    plt.ylabel("workload")
    # plt.plot(buff)
    # plt.plot(LDMResult)

    plt.show()
