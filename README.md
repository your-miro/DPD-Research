# Behavioral Modeling of GaN Power Amplifiers Using BiLSTM & DPD with Adaptive Cost Functions {Under Maintenance}

## Overview

This repository presents a lightweight yet effective behavioral modeling pipeline for GaN-based Power Amplifiers (PAs) using TensorFlow and Keras. The work explores whether different GaN PAs can be matched in behavior through minimal-complexity BiLSTM architectures and gain adjustment, alongside introducing a novel dual-objective cost function for Digital Predistortion (DPD) training.

## Objectives

- Model the nonlinear behavior of **two GaN PAs** using BiLSTM-based RNNs with **minimal complexity**.
- Use **I/Q stream arrays** with a specified **memory depth** as inputs to the models.
- Evaluate whether **gain-staged DNNs**, applied before or after the BiLSTM models, can **match the output behavior** between the two amplifiers.
- Introduce a **custom training objective** for DPD that:
  - Prioritizes **NMSE** during the early training epochs.
  - Gradually shifts focus toward minimizing **ACLR**, using a fast FFT-based implementation in native Keras.
- Normalize outputs based on **small-signal gain (SSG)** to isolate and model only the **nonlinear components** of amplifier behavior.

## Methodology

### Data Processing

- **Input format**: Complex I/Q streams modified with time-aligned memory structures.
- **Normalization**: Each amplifier’s output is normalized by removing its **small-signal gain** (e.g., 21 dB becomes `-21 dB` gain-adjusted), to ensure the model focuses on nonlinear distortion, not absolute gain.
- **SSG Estimation**: Small-signal gain is estimated using a **dynamic moving average technique**, as detailed in [REFERENCE TO PAPER].

### Modeling

- **BiLSTM RNNs** were chosen for their ability to capture time-domain memory effects with low parameter overhead.
- A **simple feedforward DNN** is optionally added before or after the BiLSTM stage to act as a **gain alignment** or transformation layer between the PAs.
- The models are trained to **mimic the dynamic behavior** of one amplifier using data from the other, under various gain-stage configurations.

### Loss Function

A custom composite cost function was developed to enhance learning efficiency and task specificity:

```python
Loss = w(t) * ACLR + (1 - w(t)) * NMSE
w(t) = init_w + grthfact * t
```

Where:
init_w is the initial weight of the ACLR loss
grthfact is the growth factor that this weight increases by in every epoch
