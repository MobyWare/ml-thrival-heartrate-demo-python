# Thrival 2016 Demo - ML in Healthcare - Heart Rate Classification & Clustering

This is the code associated with a presentation I did at Thrival 2016. Thrival is an art and innovation festival held in Pittsburgh, PA. You can find more details on our presentation [here](http://sched.co/8Bxs) and about the Thrival festival [here](http://www.thrivalfestival.com/). [Joseph Wright](https://github.com/joegle) and I presented some of our ideas on how ML can be approachably used in healthcare. Joe is experienced in visualizaitons and digital signal processing (DSP) and I have experience in machine learning.

## Goal
I want to see how feasible it is to use simple ML techniquest to identify someone from their heart rate. I worked with my colleague Joseph to test this theory.

## Overview
We are going to use clustering and discrete signal processing to help classify heart-rates. The gist is I took heart rate data that Joe measured using his arduino project [here](https://github.com/joegle/hrv-biofeedback). We also used some of his experiments with fast fourier transforms (FFT). These can reduce over a thousand samples over 20 minutes of measurements to just over a hundred touples over the frequency spectrum. We then cluster the spectra for each heart rate to see if only heart-rates from common individuals are labelled in the same cluster.

## Set up

TBD

