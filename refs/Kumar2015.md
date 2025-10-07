# DistancePPG: Robust non-contact vital signs monitoring using a camera
*Kumar M, Veeraraghavan A, Sabharwal A. DistancePPG: Robust non-contact vital signs monitoring using a camera. Biomed Opt Express. 2015 Apr 6;6(5):1565-88. doi: 10.1364/BOE.6.001565. PMID: 26137365; PMCID: PMC4467696.*

> In this paper, we address the challenge of reliable vital sign estimation for people having darker skin tones, under low lighting conditions and under different natural motion scenarios to expand the scope of camerabased vital sign monitoring.
> 

**Objectives**

- address darker skin tones
- adress low lighting conditions
- consider different natural motion scenarios

**Contributions: distancePPG algorithm**

- combine signals from different regions of the face with weighted average
- automatic determination of weights based on light intensity and blood perfusion of each region
- face tracking method based on deformable face tracker and  KLT (KanadeLucas-Tomasi) feature tracker

Testing scenarios: reading content on computer screen, watching video and talking

## Prior work

> The fact that the green channel perform better is expected as the absorption spectra of hemoglobin (Hb) and oxyhemoglobin (HbO2), two main constituent chromophores in blood, peaks in the region around 520−580 nm, which is essentially the passband range of the green filters in color cameras
> 
- Green channel performs better than red and blue for detecting pulse and respiratory rates
- Cyan, orange and green (COG) can work better than RGB

> One possible explanation for better performance of cyan-orange-green (COG) channels could be the higher overlap between the passband of cyan (470 − 570 nm), green (520 − 620 nm), and orange (530 − 630 nm) with the peak in the absorption spectra of Hb and HbO2 (520−580 nm).
> 
- face tracking
    - automatic face detection in consecutive frames presents difficulty under motion due to false negatives
    - 2D shifts computation does not capture natural motions properly like tilting or turning the face

## Background and problem definition

**Definitions**

- PPG signal $p(t)$
- Intensity signal $V(x, y, t)$
- Intensity of ilumination $I(x, y, t)$
    - intensity of ambient or any dedicated light falling on the face
    - generally assumed constant over the PPG estimation window
- Reflectance of the surface (skin) $R(x, y, t)$
    - fraction of light reflected back from the skin
    - two levels: surface reflection and subsurface reflection or backscattering
- Intensity signal can be decomposed into intensity o filumination and refllectance of the surface:
    - $V(x, y, t) = I(x, y, t) R(x, y, t)$
- Skin’s bidirectional reflectance distribution function (BRDF)

> A large part of the light incident on face gets reflected back from the surface of the skin, and is characterized by the skin’s bidirectional reflectance distribution function (BRDF). Remaining part of the incident light goes underneath the skin surface and is absorbed by the tissue and the chromophores (Hb, HbO2) present in blood inside arteries and capillaries. The volume of blood in the arteries and capillaries changes with each cardiac cycle and thus the level of absorption of light changes as well. Since PPG signal, by definition, is proportional to this cardio-synchronous pulsatile blood volume change in the tissue [13], one can estimate PPG signal p(t) by estimating these small changes in subsurface light absorption. Thus, the camerabased PPG signal is estimated by extracting small variations in the subsurface component of skin reflectance R(x, y,t).
> 
- Light intensity is assumed constant, so changes in intensity of the reflected light are proportional to changes in teh reflectance of the skin surface
    - Normally the changes in recorded intensity are dominated by changes in surface reflection unrelated to PPG
    - Spatial average of intensity level over all pixels in the face region
        - incoherent changes will cancel out
- Spatially averaged intensity signal is proportional to PPG signal
- Filters generally between 0.5 Hz - 5 Hz

### Key challenges

**Very low signal strength**

> PPG signal extracted from camera video have low signal strength. This is because the skin vascular bed contains a very small amount of blood (only 2−5% of total blood in the body), and the blood volume itself experiences only a small (5%) change in sync with the cardiovascular pulse [14]. Thus, the change in subsurface skin reflectance due to cardio synchronous changes in blood volume is very small. The small change in subsurface reflectance results in very small change in light intensity recorded using a camera placed at a distance.
> 
- Strength of PPG from a patch of the skin surface depends on:
    - intensity of light incident on that patch
    - total amount of blood volume change underneath that patch
- Including regions of the face with limited blood perfusion in the extraction of the PPG signal can increase the noise
- Necessity to combine PPG signal obtained from different regions to maximize SNR

**Motion artifact** 

- Tracking non-rigid regions (cheeks for ex) of the face independently is ideal
- Changes in facial regions lead to changes in incident light intensity of each region
- small changes in light direction can lead to large changes in skin surface reflectance which is characterized by the highly non-linear BRDF of the skin surface
- Regions that are useful to PPG estimation do not present great changes in magnitude throughout the PPG estimation window

### DistancePPG algorithm

- Maximum ratio combining
    - combine average pixel intensity signal of different regions to improve overall SNR
- Region based motion tracking

**Step 1**: Input: Green channel video of a person, landmark points around eyes, nose, mouth in face detected

**Step 2**: Face is divided into seven regions, each region tracked using KLT, motion modeled using rigid affine fit

- ROIs are small enough si that the blood perfusion within them can be assumed constant
- Spatial ilumination variation inside a ROI is assumed constant

**Step 3**: Each tracked region is divided into 20x20 pixel block, avg. pixel intensity yi(t) computed from each ROI

**Step 4**: Goodness metric Gi computed for each ROI, overall camera PPG estimated using weighted average

**Modeling**

$$
y_i(t) = I_i \Big (\alpha_i \cdot p(t) + b_i \Big ) + q_i(t)
$$

- $y_i(t)$: measurements
- $I_i$: incident light intensity in ROI i
- $\alpha_i$: strength of blood perfusion
    - strength of modulation of light backscattered from the subsurface due to the pulsatile blood volume change
    - varies over skin regions
    - varies for each individual
    - depends on the wavelength of incident light (since Hb and HbO2 absorption depends on wavelength of light)
- $b_i$: surface reflectance from the skin the region i
- $q_i$: camera quantization noise

**MRC (Maximum ratio combining) algorithm**

- Compute $y_i(t)$ for all ROIs
- Filter using a bandpass filter (0.5 - 5 Hz)
    - filtered signal: $\hat{y}(t)$
    - $\hat{y}_n(t) = A_n p(t) + w_n(t)$
    - A is the strength of the PPG signal in region i
    - $w$ is the noise component due to camera quantization
- Combine different signals io each ROI with a weighted average:
    
    $$
    \hat{p}(t) = \sum_{i=1}^n G_i \cdot\hat{y}_i(t)
    $$
    
- Weights determined based on maximum ratio diversity

$$
G_i = \frac{A_i}{|| w_i(t) ||^2}
$$

- $G_i$ as goodness metric for region i
- Delays between blood perfusion of each region are ignored
    - 10 ms compared to 30 - 60 fps of the camera
- Rejection of regions with unusual large signals
    - These variations are often due to illumination change or motion artifacts
    - Reject regions with amplitude greater than a threshold $A_{th}$
- Final estimate of the PPG signal

$$
\hat{p}(t) = \sum_{i=1}^{n}G_i \hat{y}_i(t) I(\hat{y}_{max, i} - \hat{y}_{min, i} < A_{th})
$$

where the indicator function I gives the maximum value for $\hat{y}_i(t)$ during a T seconds duration

**Goodness metric Gi**

- Estimate the goodness metric as a ratio of the power of recorded $\hat{y}(t)$ around the pulse rate (PR) to the power of the noise in the passband of the filter.

$$
G_i(PR) = \frac{    \int_{PR-b}^{PR+b} \hat{Y}_i(f)\, df}{    \int_{B_1}^{B_2} \hat{Y}_i(f)\, df     -     \int_{PR-b}^{PR+b} \hat{Y}_i(f)\, df}
$$

where $\hat{Y}(f)$ is the power spectral density of the recorded signal $\hat{y}(t)$, PR is the pulse rate and B1 and B2 are the passband filter frequencies.

- As $G_i$ depends on the unknown pulse rate, it can be assumed as 1 as a first step to estimate the PPG signal. The peak in the spectrum of this estimated PPG is then used as PR
- Higher goodness metric are in regions with higher strength of the PPG signal relative to noise power
- Forehead and cheek have better goodness metrics
- Eyes have low Gi because of eye movement which make the signal amplitude greater than the threshold

**Region based motion tracking algorithm**

- necessity to keep track of each ROI
- Landmark location
- Definition of 3 regions on the forehead, one region for each cheek and 2 regions on the chin
    - exclusion of regions with large motion artifacts (eyes, mouth)
- Identification of 50 feature points inside each region
    - J. Shi and C. Tomasi, “Good features to track,” in “, 1994 IEEE Computer Society Conference on Computer
    Vision and Pattern Recognition, 1994. Proceedings CVPR ’94,” (1994), pp. 593–600.
- KLT feature tracker to track features across the video
    - B. D. Lucas and T. Kanade, “An iterative image registration technique with an application to stereo vision,”
    Proceedings DARPA Image Understanding Workshop (1981), pp. 674–679.
    - C. Tomasi and T. Kanade, “Detection and tracking of point features,” Technical Report MU-CS-91-132, Carnegie
    Mellon University (1991).
- Computation of affine fit for each region using the feature points
    - Automatic detection of feature tracking failure
    - Z. Kalal, K. Mikolajczyk, and J. Matas, “Forward-backward error: Automatic detection of tracking failures,” in
    “2010 20th International Conference on Pattern Recognition (ICPR),” (2010), pp. 2756–2759.
- random sample consensus (RANSAC) to compute an estimation of the affine fit by rejecting outliers
    - M. A. Fischler and R. C. Bolles, “Random sample consensus: A paradigm for model fitting with applications to
    image analysis and automated cartography,” Commun. ACM 24, 381–395 (1981)
- the 7 planar regions of the face are divided in PPG ROIs

```jsx
procedure TRACKPPGREGION(V(x, y, t))           ▷ Track ROIs given a sequence of frames
    t ← 0                                      ▷ Current Frame Counter
    while t ≤ T_total do                       ▷ T_total is total number of frames
        if t mod (T × FPS) = 0 then            ▷ Restart after every T sec. FPS is camera frame rate
            LP ← DEFORMABLEFACE(V(x, y, t))    ▷ LP are landmark points
            {𝒫}_t ← DEFINEPLANERREGION(LP)     ▷ {𝒫}_t are planar regions at time t
            {ℛ}_t ← DEFINEPPGREGION(P_t)       ▷ {ℛ}_t are the PPG ROIs for tᵗʰ frame
            GF₀ ← GOODFEATURES(V(x, y, t), P_t)▷ good features inside each region
        else
            GF₁ ← TRACKERKLT(V(x, y, t), V(x, y, t−1), GF₀)
            AM_t ← AFFINERANSAC(GF₁, GF₀)
            ▷ AM denotes the affine model parameter computed separately for each planar region
            {𝒫}_t ← {𝒫}_{t−1} × AM_t
            {ℛ}_t ← {ℛ}_{t−1} × AM_tᴾ
            ▷ Each PPG region in set {ℛ} tracked using encompassing planar region motion model
            GF₀ ← GF₁
        end if
        t ← t + 1
    end while
    return {ℛ}_t                               ▷ Return tracked PPG regions {ℛ}_t locations
end procedure
```

**Implementation details**

- ROis of 20x20 pixels considering the camera resolution and distance of 0.5m of the patient
- Restart KLT ever 5-10 seconds
- RANSAC
    - tolerance of 2 pixels
    - inlier fraction of 0.7
    - maximum number of iteration N = 20
- Amplitude threshold of 8
- Recompute goodness metric after every 5-10 seconds

### Experimental setup

- Flea 3® USB 3.0 FL-U3-13E4MC monochrome camera
    - 30 FPS
    - Resolution of 1280 × 1024
    - 8 bits per pixel
    - 40 nm full-width half-max (FWHM) Green filter (FB550- 40 from Thor labs ®) in front of the monochrome camera
- Flea 3® USB 3.0 FL-U3-13E4C-C color camera
    - 30 FPS
    - RGB Bayer pattern
    - 1280×1024
    - 8 bits per pixel
- Ground truth? Texas Instruments AFE4490SPO2EVM pulse oximeter module
    - 500 Hz
- Distance camera-subject of 0.5 m
- Processing in MATLAB
- Ambient lighting of 500 lux
- Variation of skin tone, motion and ambient light intensity (50 - 650 lux)
- Pulse oximeter in the earlobe because of its proximity to the face region

### Signal to noise ratio definition

If $z(t)$ is the PPG aquired with the pulse oximeter and $n_z$ its noise, then $k(t)$ is the PPG estimated from the camera-based system and $n_k$ its noise

The noise in tha camera based system can be estimated as:

$$
n_k(t) \equiv k(t) - \frac{\langle k(t), z(t) \rangle}{\langle z(t), z(t) \rangle} \cdot z(t)
$$

The underlying signal is estimated as

$$
s_k(t) \equiv \frac{\langle k(t), z(t) \rangle}{\langle z(t), z(t) \rangle} \cdot z(t)
$$

The SNR would, then, be:

$$
\mathrm{SNR} \equiv \frac{\| s_k(t) \|^2}{\| n_k(t) \|^2}
$$

### Quantification of physiological parameters

- pulse rate
    - spectrum (FFT) of the PPG signal over every 10 seconds
    - hamming window of 10 seconds
    - 5 seconds overlap
    - PR is the highest power in the spectrum
- PRV (or interbeat interval IBI)
    - interpolate the camera-based PPG signal with a cubic spline to a frequency of 500 Hz
    - custom algorithm for peak detection
    - time interval between peaks is the PRV

## Results

- distancePPG had increased SNR specially for darker skin tones and better agreement in the BlandAltman plot compared to face averaging methods
- distancePPG also improves SNR for all motion scenarios
- results similiar to ICA method in static scenarios, but distancePPG shows a slight improvement in SNR

## Discussion

- goodness metric as good substitute for SNR as signal quality index of the PPG signal

> Figure 3 of goodness metric overlay clearly shows that forehead region is best for extracting PPG signal from the face. Our finding is in agreement with the findings of other researchers that forehead represent a suitable site for camera-based PPG estimation. This is because the forehead region have much better blood perfusion compared to other regions
> 
- goodness metric as indicator of blood perfusion
    - M. Fernandez, K. Burns, B. Calhoun, S. George, B. Martin, and C. Weaver, “Evaluation of a new pulse oximeter sensor,” American Journal of Critical Care: An Official Publication, American Association of Critical-Care
    Nurses 16, 146–152 (2007).
- One solution to further reduce the RMSE of PRV is to use higher frame rate camera (e.g.,100 fps)