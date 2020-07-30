# MAIN script for running the FBC (Forecast Based Classification) model

from FBC import FBC

M = 10  # the no. of trained <LSTM> networks for each class

instance_FBC = FBC(seq_len=24, M=M)




