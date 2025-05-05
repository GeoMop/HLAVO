# hlavo-station
ESP32-S3 DevKit source codes for HLAVO project (meteo station and column experiment).

This repository is structured so that it could be used both wit ArduinoIDE and Platformio extension in Visual Studio Code.

The directory `examples` includes all main scripts (`.ino` files).
The main scripts are
- `hlavo-station` for meteo station,
- `hlavo-column` for column experiment.
All other scripts are tests and intermediate steps towards final solutions.

The directory `libraries` includes all used external libraries.
Directory name `examples` and `libraries` are due to ArduinoIDE, for Platformio it is set in `platformio.ini`.
Some reusable parts of code developed for HLAVO project are gathered in `libraries/Hlavo`.
Most of the libraries could be added as Git submodules if necessary,
see the dependencies - their versions and github links in `libraries/Hlavo/library.json`.

Once the VS Code, Platformio extension and the COM port is set up, check `platformio.ini`
whether the correct upload COM port is defined, e.g. `/dev/ttyACM0` in Ubuntu.
Then select (uncomment) prefered `src_dir` and build the selected application.
Once connected to a microchip module; monitor, build and upload the application using Platformio toolbar.
