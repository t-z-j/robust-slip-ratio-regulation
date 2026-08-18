# Robust Slip-Ratio Regulation for Agricultural Tractors with Transformer-Enhanced Speed Estimation

This repository contains the source code associated with the article:

**"Robust Slip-Ratio Regulation for Agricultural Tractors with Transformer-Enhanced Speed Estimation"**

## Repository Structure

### 1_ASTSMC_Controller_Source_Code

This folder contains the MATLAB implementation of the boundary-layer adaptive super-twisting sliding-mode controller (ASTSMC) for four-wheel-independent tractor slip-ratio regulation.

Main file:

- `Tractor_ASTSMC_Control.m`

#### Description and Usage

The controller receives the slip ratios of the four driven wheels, the corresponding longitudinal wheel-force signals, and the external traction-load signal. Based on these inputs, the ASTSMC algorithm calculates the drive-torque commands for the four independently driven wheels to regulate the wheel slip ratios toward the desired reference value.

The controller can be called in MATLAB as:

```matlab
[T_L1, T_L2, T_R1, T_R2] = Tractor_ASTSMC_Control( ...
    lambda_L1, lambda_L2, lambda_R1, lambda_R2, ...
    Fx_L1, Fx_L2, Fx_R1, Fx_R2, F_load);
