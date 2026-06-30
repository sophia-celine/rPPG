# Assessment of remote photoplethysmography algorithms for vitals signs estimation in Intensive Care Unit patients

_Validação de algoritmos de fotopletismografia remota para estimativa de sinais vitais em pacientes de Unidades de Terapia Intensiva_

Repository for the development of research on the validation of remote photoplethysmography (rPPG) algorithms for vital signs estimation in Intensive Care Unit patients for the Biomedical Engineering program in Electrical Engineering at the Polytechnic School of the University of São Paulo.

Repositório para desenvolvimento de pesquisa de validação de algoritmos de fotopletismografia remota em unidades de terapia intensiva para o programa de Engenharia Biomédica do [Programa de Pós-Graduação em Engenharia Elétrica](https://ppgee.poli.usp.br/pb) da Escola Politécnica da Universidade de São Paulo.

_**Abstract** - Vital sign monitoring is essential for evaluating the physiological state of a patient and is typically performed using contact-based medical devices. Nonetheless, non-invasive monitoring approaches offer relevant benefits in both clinical and non-clinical settings by increasing patient comfort, reducing contact-associated risks, and providing greater flexibility of application. In this context, remote photoplethysmography (rPPG) has emerged as a promising technique for estimating physiological parameters such as heart rate, respiratory rate, oxygen saturation and blood pressure by detecting cyclical variations of the cardiovascular system which can be extracted from subtle changes in the color intensity of pixels from the skin region captured by video. Although several methods have been proposed, including unsupervised approaches and deep learning–based techniques, most studies are conducted under controlled conditions and involve predominantly healthy individuals. In view of this limitation, this work aims to assess the performance of the main rPPG algorithms using video recordings of patients in a public Intensive Care Unit (ICU) which is part of the brazilian Unified Health System (Sistema Único de Saúde - SUS). In addition, a new dataset specific to this context is proposed, comprising video recordings acquired using an industrial video camera and reference signals collected directly from bedside multiparameter monitors. The results aim to assess the robustness and clinical feasibility of rPPG methods under realistic intensive care conditions._

# Data acquisition

Videos are acquired from ICU patients using an industrial VCXG.2-57C video camera. Ground truth ECG, PPG and respiratory signals are recorded from bedside monitors (BSM-3000 Series, Nihon Kohden, Japan).  The ECG was recorded with a sample rate of 250 Hz, respiratory signal was recorded with sample rate of 125 Hz and the PPG signal was recorded at 62,5 Hz.

# Video processing

A total of 7 unsupervised algorithms (GREEN, ICA, PBV, CHROM, POS, OMIT and LGI) and 5 deep learning-based models (DeepPhys, PhysNet, TSCAN, PhysFormer and EfficientPhys) were used to obtain the blood volume pulse (BVP) waveform, also denoted as the rPPG signal, using the [rPPG-Toolbox](https://github.com/sophia-celine/rPPG-Toolbox). HR was estimated through spectral analysis in 15-second windows and RR was estimated in 30-second windows using the amplitude, baseline and frequency respiratory-induced variations of the BVP. HR and RR estimation was evaluated using Mean Absolute Error (MAE), Root Mean Square Error (RMSE) and Mean Absolute Percentage Error (MAPE). BVP quality was evaluated using Signal-to-Noise Ratio (SNR) and cross-correlation with the ground-truth PPG. 


