from fbprophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

def analyseCSV(df):
    prophet = Prophet()
    prophet.fit(df)
    future = prophet.make_future_dataframe(periods=365)
    forecast = prophet.predict(future)
    prophet.plot(forecast)
    plt.xlabel("time")
    plt.ylabel("workload")
    plt.show()


def getLoss(yhat, ):
    pass

if __name__ == "__main__":
    df = pd.read_csv('example_wp_log_peyton_manning.csv')
    analyseCSV(df)
