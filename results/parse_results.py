import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import os

def create_csv(filename, directory, old_folder, server):
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
    new_df.columns = ['accu_tra', 'roc_tra', 'f1_tra', 'prec_tra', 'rec_tra', 'accu_test', 'roc_test', 'f1_test', 'prec_test', 'rec_test', 'time']
    better_name = filename.replace(server, "")
    better_name = better_name.replace(".txt", "")
    better_name = better_name.replace(old_folder, "")
    new_df.to_csv(directory+"/"+better_name+".csv")

if __name__ == "__main__":
    try:
        experiment = sys.argv[1]
        model = sys.argv[2]
        folder = sys.argv[3] if len(sys.argv) >= 4 else ""
        server = sys.argv[4] if len(sys.argv) >= 5 else "DockerServer"
        percentages = sys.argv[5] if len(sys.argv) >= 6 else "0.01,0.1"
        threads = sys.argv[6] if len(sys.argv) >= 7 else "1,5,15,25,35,39"
        threads = threads.split(',')
        percentages = percentages.split(',')
        files = []
        for thread in threads:
            files.extend([folder+server+model+"_errorpond_" + experiment + str(thread) + "_"+str(percentage)+".txt" for percentage in percentages])
        if not os.path.exists(server+"_"+experiment+"_"+model):
            os.makedirs(server+"_"+experiment+"_"+model)
        print "USING DATASET: " + experiment
        print "USING MODEL  : " + model
        print "USING SERVER : " + server
        print "USING THREADS: " + str(threads)
        print "USING %      : " + str(percentages) + "\n"
        print "PROCESSING FILES...\n"
        for file in files:
            print "PROCESSING FILE: " + file
            create_csv(file, server+"_"+experiment+"_"+model, folder, server)
        print "\nALL FILES WERE STORED AT: " + server + "_" + experiment + "_" + model + "/ FOLDER\n"
        print "DONE."
    except IndexError, e:
        print "Usage: python parse_results.py dataset model [folder] [server] [percentages] [threads]\n"
        print "     Example 1: python parse_results.py astro DT => This assumes 1,5,15,25,35,39 threads, 0.01,0.1 percent, actual folder and DockerServer server."
        print "     Example 2: python parse_results.py astro DT \"../ASTRODATA/\" ColosoOfi \"0.01,0.5\" \"1,15,39\" => Only uses threads 1,15 and 39, 0.01 and 0.5 percent, ASTRODATA/ folder and ColosoOfi server."

