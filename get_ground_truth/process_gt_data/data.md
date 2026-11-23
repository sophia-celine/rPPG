# Data description 

## Reference signals

Data were acquired from ICU patients in a public hospital in São Paulo, Brazil, between August and November of 2026. The signals were acquired from the
bedside monitors (BSM-3000 Series, Nihon Kohden, Japain). Not all signals were available for all patients. Below is described all possibly available signals.

- ECG
    - Waveforms (sample rate: 250 Hz, digital signals)
        - lead I
        - lead II
        - lead III
        - aVR
        - aVL
        - aVF
        - 2 precordial leads Vn1 and Vn2 (V1 to V6, not specified)
    - Heart rate (1 Hz, bpm)

- Respiration
    - Waveforms (sample rate: 125 Hz)
        - Impedance signal waveform (digital signal)
        - CO2 pressure waveform (mmHg)
    - Respiratory rate from impedance signal waveform (1 Hz, rpm)  
    - Respiratory rate from CO2 pressure waveform (1 Hz, rpm)

- Blood pressure
    - Waveforms (sample rate: 125 Hz)
        - Invasive arterial pressure waveform (mmHg)
    - Non-invasive systolic blood pressure (1 Hz, mmHg)
    - Non-invasive diastolic blood pressure (1 Hz, mmHg)
    - Non-invasive mean blood pressure (1 Hz, mmHg)

- Oxygen saturation
    - Waveforms (sample rate: 62,5 Hz)
        - Pulse oximeter plethysmography waveform (digital signal)
    - Oxygen saturation (1 Hz, %)