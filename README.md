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

## Results

Results are applicable to single, and multiple channel FDM-ed signals.

Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer                                ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ bidirectional                        │ (None, 6, 64)               │           8,960 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ global_max_pooling1d                 │ (None, 64)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense                                │ (None, 2)                   │             130 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 27,272 (106.54 KB)
 Trainable params: 9,090 (35.51 KB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 18,182 (71.03 KB)

# (Training PA1 & PA2)
![image](https://github.com/user-attachments/assets/fba54d4a-280a-40aa-978b-3844b54142b9)

Behavioral modelling of PA1 nmse is -33.14927716768715 dB

![image](https://github.com/user-attachments/assets/e1409790-d6a0-4f86-93fb-5835f9fbc873)

Behavioral modelling of PA2 nmse is -33.61280015701266 dB

# (DPD using Loss Function)
![image](https://github.com/user-attachments/assets/abc63a44-1a78-4f13-b2f5-056c7b37f19c)

Before Linearization:                              After Linearization:
Upper ACLR = -28.43972292246042 (dBc)              Upper ACLR = -44.25079531333139 (dBc)
Lower ACLR = -28.43972292246042 (dBc)              Lower ACLR = -45.03776684959236 (dBc)

# To be Continued...
