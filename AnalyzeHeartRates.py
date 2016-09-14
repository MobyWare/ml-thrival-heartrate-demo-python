import numpy as np
import scipy.signal as signal
from sklearn.cluster import AffinityPropagation
from sklearn import metrics

def fix(array):
    n = np.array([])
    for x in array:
        n = np.append(n, np.zeros((x-1)/100.0))
        n = np.append(n, [1])
    return n

def printClusterMetrics(clusterModel, observations):
    clusters = clusterModel.fit(observations)
    centers = clusters.cluster_centers_indices_
    labels = clusters.labels_
    if centers is None or len(centers) < 1:
        print("Clustering algorithm didn't converge.")
    else:        
        silhouetteMetric = metrics.silhouette_score(observations, labels, metric='euclidean')
        print(\
            'There are {} obervations and {} clusters, with a confidence of {}.\nThe labels were: {}'.format(observations.shape[0], len(centers), silhouetteMetric, labels))

samples = None
files = ['data/alice1.txt', 'data/alice2.txt', 'data/bob1.txt']

for index, filename in enumerate(files):
    intervals = np.loadtxt(filename)
    intervals = intervals[intervals < 2000] 
    frequencySpectrum, amplitude = signal.welch(fix(intervals), 5, scaling='spectrum')
    if samples is None:
        samples = amplitude[np.newaxis]
    else:
        samples = np.concatenate((samples, amplitude[np.newaxis]), axis=0)
    print (\
        '{}) {}. Both the spectrum: {} and amplitude: {} should had have the same length. The currnet # of samples is: {}.\n Amplitude (1st five): {}'.\
        format(str(index + 1), filename, len(frequencySpectrum), len(amplitude), samples.shape[0], amplitude[0:5]))

printClusterMetrics(AffinityPropagation(max_iter=5000), samples)



