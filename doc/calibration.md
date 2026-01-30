# Calibration of the models

In order to bootstrap model parameters, we apply several stages of calibration
mainly in following order:

1. Calibration of the deep model to the historical data from near by stations. Crude infiltration model.
2. Calibration of the surface model to the collected datasets using decoupled deep model for the bottm boundary condition.
3. Calibration of the composed model over historical (surface surrogate) or collected data.
4. Training a surface surrogate for data driven prediction of the infiltration from the station only data.

## Implementation (in progress)
[hlavo/calibration](../hlavo/calibration/README.md)
