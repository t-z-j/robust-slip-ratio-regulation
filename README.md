# Robust Slip-Ratio Regulation for Agricultural Tractors with Transformer-Enhanced Speed Estimation

This repository contains the source code associated with the article:

**"Robust Slip-Ratio Regulation for Agricultural Tractors with Transformer-Enhanced Speed Estimation"**

## Repository Structure

### 1_ASTSMC_Controller_Source_Code

This folder contains the MATLAB implementation of the boundary-layer adaptive super-twisting sliding-mode controller (ASTSMC) for four-wheel-independent tractor slip-ratio regulation.

Main file:

- `Tractor_ASTSMC_Control.m`

#### Usage

The controller takes the four wheel slip ratios, longitudinal wheel-force signals, and external traction-load signal as inputs, and outputs the drive-torque commands for the four independently driven wheels.

The function can be directly integrated into a MATLAB/Simulink control model and called at each control step.

### 2_Transformer_Process_Noise_Scaling_Source_Code

This folder contains the Python/PyTorch implementation of the Transformer-based process-noise scaling model used in the longitudinal-speed estimation framework.

Main file:

- `transformer_process_noise_scaling_model.py`

#### Usage

The model receives a sliding window of 100 samples with 30 input features and outputs a process-noise scaling factor `q_scale`.

The generated `q_scale` is supplied to the external EKF algorithm to adapt the process-noise covariance during longitudinal-speed estimation.

This file contains the Transformer-based process-noise scaling model only and does not include the complete EKF algorithm.

## Software

- MATLAB/Simulink
- Python
- PyTorch

## Availability

The source code is provided to support the reproducibility and transparency of the associated research.
