import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from sklearn.cluster import AffinityPropagation
from sklearn import metrics

def fix(array):
    n = np.array([])
    for x in array:
        n = np.append(n, np.zeros((x-1)/100.0))
        n = np.append(n, [1])
    return n

def printClusterMetrics(affinity_cluster, observations):
    centers = affinity_cluster.cluster_centers_indices_
    labels = affinity_cluster.labels_
    if centers is None or len(centers) < 1:
        print("Clustering algorithm didn't converge.")
    else:        
        silhouetteMetric = metrics.silhouette_score(observations, labels, metric='euclidean')
        print(\
            'There are {} obervations and {} clusters, with a confidence of {}.\nThe labels were: {}'.\
            format(observations.shape[0], len(centers), silhouetteMetric, labels))


def getAverageBPM(samples):
    bpm_values = np.array([])
    TOP_N = 5

    for observation in samples:
        bpm_values = np.append(bpm_values, np.average(map(lambda x: 60000/x, observation)))
    
    if bpm_values.size <= TOP_N:
        print('The beats per minute were: {}.'.format(bpm_values))
    else:
        print('The beats per minute were (1st 5 of {}):{}.'.format(bpm_values[0:TOP_N]))
    
    return bpm_values

def plotFFTofHeartRate(amplitudes, frequencySpectrum, seriesLabels=None, clusterLabels=None):
    labels = None
    default_colors = np.array(['red', 'black', 'green', 'blue', 'yellow'])
    unique_cluster_labels = None
    cluster_colors = None
    cluster_labels = None
    temp_cluster_labels = None

    if seriesLabels is None:
        labels = np.array(["label"+i for i in range(0, observations.shape[0])])
    else:
        labels = seriesLabels
       
    if not (clusterLabels is None):
        if clusterLabels.size < amplitudes.shape[0]:
            raise ValueError("There are too few labels for the number of observations")

        if clusterLabels.size > default_colors.size:
            raise RuntimeError("We have too many labels we only suport {} clusters in this version.".format(default_colors.size))
        
        unique_cluster_labels = np.unique(clusterLabels)
        cluster_colors = {unique_cluster_labels[i]:default_colors[i] for i in range(0,unique_cluster_labels.size)}
        cluster_labels = {i:None for i in unique_cluster_labels}

        for idx, lbl in enumerate(labels):
            if cluster_labels[clusterLabels[idx]] is None:
                cluster_labels[clusterLabels[idx]] = lbl
            else:
                cluster_labels[clusterLabels[idx]] += " | " + lbl
        
        labels = [cluster_labels[clusterLabels[i]] for i in range(0, labels.size)]
    
        for index, amp in enumerate(amplitudes):
            plt.plot(frequencySpectrum, amp, label=labels[index], color=cluster_colors[clusterLabels[index]])
            plt.grid(True)
    
        print('Starting plot with clusters ...')
    else:
        for index, amp in enumerate(amplitudes):
            plt.plot(frequencySpectrum, amp, label=labels[index])
            plt.grid(True)        
        
        print('Starting plot wihtout clusters ...')

    plt.legend()
    plt.show()

    
samples = None
intervals_collection = None
affinity_cluster = None
MAX_INTERVAL_NUMBER = 1000
MAX_INTERVAL_VALUE = 2000
interval_number = MAX_INTERVAL_NUMBER

files = np.array(['data/alice1.txt', 'data/alice2.txt', 'data/bob1.txt', 'data/terry_1.txt', 'data/terry_2.txt'])

for index, filename in enumerate(files):
    intervals = np.loadtxt(filename)
    intervals = intervals[intervals < MAX_INTERVAL_VALUE] 
    frequencySpectrum, amplitude = signal.welch(fix(intervals), 5, scaling='spectrum')
    if intervals.size > MAX_INTERVAL_NUMBER:
        interval_number = MAX_INTERVAL_NUMBER
    else:
        interval_number = intervals.size


    if samples is None:
        samples = amplitude[np.newaxis]
        intervals_collection = intervals[0:interval_number][np.newaxis]
    else:
        samples = np.concatenate((samples, amplitude[np.newaxis]), axis=0)
        intervals_collection = np.append(intervals_collection, intervals[0:interval_number][np.newaxis], axis=0)
    print (\
        '{}) {}. Both the spectrum: {} and amplitude: {} should had have the same length. The currnet # of samples is: {}.\n Amplitude (1st five): {}'.\
        format(str(index + 1), filename, len(frequencySpectrum), len(amplitude), samples.shape[0], amplitude[0:5]))

affinity_cluster = AffinityPropagation().fit(samples)

getAverageBPM(intervals_collection)
#printClusterMetrics(affinity_cluster, samples)
#plotFFTofHeartRate(samples, frequencySpectrum, files)

