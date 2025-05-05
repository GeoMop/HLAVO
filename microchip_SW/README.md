# hlavo-station
ESP32-S3 DevKit source codes for HLAVO project (meteo station and column experiment).

This repository is structured so that it could be used both wit ArduinoIDE and Platformio extension in Visual Studio Code.

The directory `examples` includes all main scripts (`.ino` files).
The main scripts are
- `hlavo-station` for meteo station,
- `hlavo-column` for column experiment.
All other scripts are tests and intermediate steps towards final solutions.

The directory `libraries` includes all used external libraries.
Directory name is due to ArduinoIDE, for Platformio it is set in `platformio.ini`.
Most of the libraries could be added as Git submodules if necessary.
Some reusable code developed for HLAVO project is gathered in `libraries/Hlavo`.
