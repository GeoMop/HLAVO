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

Own code in `libraries/Hlavo` mostly includes some wrapping functionality:
- `common.h` - common Hlavo constants and functions
- `clock.h` - wrapper for RTC (real time clock)
- `file_info.h` - wrapper for file handling on SD card filesystem
- `Logger.h` - Logger prototype for writing loggin messages to a SD card
- `CSV_handler.h` - wrapper for CSV handling
- `data_base.h` - base class for CSV data
  - `bme280_data.h` - data structure for collecting data from BME280 humidity sensor to CSV
  - `meteo_data.h` - data structure for collecting meteo data from meteostation sensor to CSV
  - `pr2_data.h` - data structure for collecting data from PR2 sensor to CSV
  - `teros31_data.h` - data structure for collecting data from PR2 sensor to CSV
  - `column_flow_data.h` - data structure for collecting water height and flow data in column experiment

- `sdi12_comm.h` - wrapper for SDI12 library regarding Hlavo usage (PR2/Teros31 sensors)
- `pr2_reader.h` - wrapper for SDI12 to communicate with PR2 sensors
- `teros31_reader.h` - wrapper for SDI12 to communicate with Teros31 sensors
- `water_height_sensor.h` - ultra sound measurement of water height (uses analog reader and converts [V]->[mm])
- `weather_station.h` - wrapper for weather_meters library
