import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt


def generate_plot(measure, files, threads, model, percentage, experiment, server):
    dfs = []
    accu_df = None
    for idx, file in enumerate(files):
        dfs.append(pd.read_csv(file))
        if accu_df is None:
            accu_df = pd.concat([dfs[-1]['round']], axis = 1)
        accu_df['accu_'+str(threads[idx])] = dfs[-1][measure]
    #print accu_df
    plt.figure()
    accu_df.plot(x='round')
    #plt.show()
    plt.savefig(server + "_" + experiment + "_" + model + "/" + experiment+"_"+model+"_"+percentage+"_"+measure+'.png')
    plt.close()

if __name__ == "__main__":
    try:
        experiment = sys.argv[1]
        model = sys.argv[2]
        percentage = sys.argv[3]
        server = sys.argv[4] if len(sys.argv) >= 5 else "DockerServer"
        threads = sys.argv[5] if len(sys.argv) >= 6 else "1,5,15,25,35,39"
        threads = threads.split(',')

        files = [server + "_" + experiment + "_" + model + "/" + model+"_errorpond_" + experiment + str(thread) + "_"+str(percentage)+".csv" for thread in threads]
        measures = ['accu_test', 'roc_test', 'f1_test', 'prec_test', 'rec_test', 'time']

        print "USING DATASET: " + experiment
        print "USING MODEL  : " + model
        print "USING SERVER : " + server
        print "USING THREADS: " + str(threads)
        print "USING %      : " + percentage + "\n"
        print "CREATING PLOTS...\n"
        for measure in measures:
            generate_plot(measure, files, threads, model, percentage, experiment, server)
        print "ALL PLOTS WERE STORED AT: " + server + "_" + experiment + "_" + model + "/ FOLDER\n"
        print "DONE."
    except IndexError, e:
        print "Usage: python create_plots.py dataset model percentage [server] [threads]\n"
        print "The script will look for files on folder dataset_model/ and will create plots for measures accuracy, ROC, F1, precision, recall and time on the test set."
        print "\n     Example 1: python create_plots.py astro DT 0.01 => This assumes 1,5,15,25,35,39 threads and DockerServer server"
        print "     Example 2: python create_plots.py astro DT ColosoOfi 0.01 \"1,15,39\" => Only uses threads 1,15 and 39 and ColosoOfi server"
