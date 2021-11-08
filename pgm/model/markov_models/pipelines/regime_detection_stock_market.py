import datetime
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import prefect
from matplotlib.dates import YearLocator
from pandas_datareader import data
from prefect import Flow, task

from pgm.model.markov_models.hmm.inference import (
    compute_mean_and_vars_hidden_state,
    decode_hidden_states_time_series,
)
from pgm.model.markov_models.hmm.train import (
    plot_trained_parameters,
    train_gaussian_hmm,
)

ticker = "TSLA"
start_date = datetime.date(2003, 7, 1)
end_date = datetime.date(2017, 7, 1)

logger = prefect.context.get("logger")


@task
def get_quotes_data_finance(name="yahoo"):
    Stocks = data.DataReader(ticker, name, start_date, end_date)
    Stocks.reset_index(inplace=True, drop=False)
    Stocks.drop(["Open", "High", "Low", "Adj Close"], axis=1, inplace=True)
    Stocks["Date"] = Stocks["Date"].apply(datetime.datetime.toordinal)
    d = os.path.join(Path().resolve().parents[3], "data/stocks.csv")
    Stocks.to_csv(d)
    return Stocks


@task(nout=3)
def process_and_plot_stocks_data(Stocks):
    Stocks = list(Stocks.itertuples(index=False, name=None))
    dates = np.array([q[0] for q in Stocks], dtype=int)
    end_val = np.array([q[1] for q in Stocks])
    volume = np.array([q[2] for q in Stocks])[1:]
    diff = np.diff(end_val)
    dates = dates[1:]
    end_val = end_val[1:]
    X = np.column_stack([diff, volume])
    plt.figure(figsize=(15, 5), dpi=100)
    plt.title(ticker + " - " + end_date.strftime("%m/%d/%Y"), fontsize=14)
    plt.gca().xaxis.set_major_locator(YearLocator())
    plt.plot_date(dates, end_val, "-")
    plt.show()
    return X, dates, end_val


def main():
    with Flow("hmm-stocks") as flow:
        stocks = get_quotes_data_finance()
        X, dates, end_val = process_and_plot_stocks_data(stocks)
        model = train_gaussian_hmm(X)
        hidden_states = decode_hidden_states_time_series(X, model)
        compute_mean_and_vars_hidden_state(model)
        plot_trained_parameters(model, hidden_states, dates, end_val)

    flow_state = flow.run()
    flow.visualize(flow_state=flow_state)


if __name__ == "__main__":
    main()
