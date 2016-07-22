import pandas as pd
import sys
import numpy as np

try:
    filename = sys.argv[1]
    df = pd.read_csv(filename, sep = '&', header=None)

    # get accumulated time
    sum = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    final_sum = []
    for index, row in df.iterrows():
        if index % 20 == 0 and index != 0:
            final_sum.extend(sum)
            sum = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elindex = row[0].astype(np.int64) - 2
        if elindex == -1:
            elindex = 0
        ind = row[0].astype(np.int64) - 1
        sum[ind] = row[11] + sum[elindex]
        # special case to handle last case
        if index + 1 == len(df):
            final_sum.extend(sum)

    total_time = pd.DataFrame({'time': final_sum})
    cdf = pd.concat([df, total_time], axis=1)
    # agrupamos por rondas
    grouped_round = cdf.groupby(cdf[0])
    prom = grouped_round.mean()
    new_df = pd.concat([prom[1], prom[2], prom[3], prom[4], prom[5], prom[6], prom[7], prom[8], prom[9], prom[10], prom['time']], axis=1)
    new_df.index.names = ['round']
    new_df.columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'time']
    new_df.to_csv(filename+"_PARSED.csv")
    print "END."
except IndexError, e:
    print "You need to pass a input filename"
