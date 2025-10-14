# Measuring pulse rate with a webcam—A non-contact method for evaluating cardiac activity

*M. Lewandowska, J. Ruminski, T. Kocejko, and J. Nowak, “Measuring pulse rate with a webcam—A non-contact method for evaluating cardiac activity,” in Proc. Federated Conf. Comput. Sci. Inform. Syst., 2011, pp. 405–410.*

PCA (principal component analysis)

## Setup

- thermographic camera
    - FLIR ThermaCam SC3000
- webcam
- ECG as groundtruth
    - 400 Hz  (AsCARD MrGrey v.201, Aspel)
- Indoor measurement
- Sunlight as light source
- 30 seconds videos
- Logitech Webcam 9000 Pro
- 640x480 resolution
- 20 fps
- avi format without compression
- 1 m camera-subject distance
- 2 ROIs
    - Entire face - same coordinates in all frames
    - Rectangle on forehead
- Channels decomposed in R, G and B
    - Analyses were performed for a different channels combination: RGB, RG, GB, RB

## PCA

- Commonly used for data reduction in statistical pattern recognition and signal processing
- Highlights data similarities and differences
- Find components $s_1, s_2, ... s_N$ that relate to the maximum amount of variance possible by N linearly transformed components
- Principal components are $s_i = w_i^T \cdot x$
    - $x$  is computed using the covariance matrix $E[x x^T] = C$
    - vectors $w_i$ are eigenvectors of $C$ (autovetores) that correspond to the N largest eigenvalues
- The components are ordered so that the first component $s_1$ points in direction where the inputs have the highest variance
- The second component is orthogonal to the first and points in the direction of highest variance when the first projection has been subtracted, and so on

## Algorithm

- Pixels of each ROIs channels are added
- Filter using a FIR bandpass filter (0.5 - 3.7 Hz) with 32 point Hamming window
- MATLAB FDATool
- ICA performed with MATLAB FastICA
- PCA with MATLAB *processpca*

## Results

- PCA had 10 times smaller computation time compared to ICA
- Mean heart rate was calculated using two methods: one based on the interval between positive slope zero-crossings of the 2nd or 3rd PC and the second using maximum of the power spectral density function
- When analyzing signals from only two channels, detection of a pulse rate was most effective when RG combination was considered
- Reducing the number of channles (RG instead of RGB) also increases the noise in the power spectrum