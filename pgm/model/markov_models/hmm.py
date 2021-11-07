import datetime
import os
import warnings
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import prefect
from hmmlearn.hmm import GaussianHMM
from matplotlib.dates import MonthLocator, YearLocator
from pandas_datareader import data
from prefect import Flow, task

warnings.filterwarnings("ignore")

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
    d = os.path.join(Path().resolve().parent.parent.parent, "data/stocks.csv")
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


@task(nout=2)
def run_gaussian_hmm(X, components=4, iter=1000):
    logger.info("fitting to HMM and decoding ...")
    model = GaussianHMM(
        n_components=components, covariance_type="diag", n_iter=iter
    ).fit(X)
    hidden_states = model.predict(X)
    logger.info("Transition matrix:")
    print(model.transmat_)
    return model, hidden_states


@task
def compute_mean_and_vars_hidden_state(model):
    for i in range(model.n_components):
        print("{0}th hidden state".format(i))
        print("mean = ", model.means_[i])
        print("var = ", np.diag(model.covars_[i]))


@task
def plot_trained_parameters(model, hidden_states, dates, end_val):
    fig, axs = plt.subplots(model.n_components, sharex=True, sharey=True)
    colours = cm.rainbow(np.linspace(0, 1, model.n_components))
    for i, (ax, colour) in enumerate(zip(axs, colours)):
        # Use fancy indexing to plot data in each state.
        mask = hidden_states == i
        ax.plot_date(dates[mask], end_val[mask], ".-", c=colour)
        ax.set_title("{0}th hidden state".format(i))

        # Format the ticks.
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.grid(True)
    plt.show()


if __name__ == "__main__":
    with Flow("hmm-stocks") as flow:
        stocks = get_quotes_data_finance()
        X, dates, end_val = process_and_plot_stocks_data(stocks)
        model, hidden_states = run_gaussian_hmm(X)
        compute_mean_and_vars_hidden_state(model)
        plot_trained_parameters(model, hidden_states, dates, end_val)

    flow_state = flow.run()
    flow.visualize(flow_state=flow_state)
