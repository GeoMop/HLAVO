# TESTS

### hlavo-station
Main HLAVO meteostation script combining all sensors and output.

### hlavo-column
Main HLAVO column experiment script.


## Weather meters

### hlavo-station_rain_counter
Minimalistic test. Reading rain gauge pin and getting ticks.

### hlavo-station_speed_counter
Minimalistic test. Reading anemometer pin and getting ticks.

### hlavo-station_weather
Weather station test.
- reading anemometer (ticks)
- reading rain gauge (ticks)
- reading wind direction
- uses WeatherMeters libs with several fixes (debouncing, ticks , widn direction table)

### hlavo-station_SD
Minimalistic test. Reading and writing to a file on SD card.

### hlavo-station_weather_calib
Auxiliary weather station test for calibration of wind speed and raingauge.

### hlavo-station_weather_full
Weather station full working test.
- adds real time clock (RTC)
- adds light sensor BH1750
- adds battery voltage
- adds MeteoData
- adds SD card writer
- moves some code to lib "Hlavo"


## PR2 sensors

### Arduino_SDI12_PR2
Minimalistic test. SDI12 communication with PR2 on Arduino

### hlavo-PR2
Minimalistic test. Communication with PR2 using Arduino SDI12 library.
- works fine with Arduino
- can get sensor info
- can send measure request with responce
- does not receive any data after data request
- problems caused by randomly receiving null character (solved in `hlavo-station`)

### hlavo-PR2-ESP32
Minimalistic test. Communication with PR2 using ESP32-SDI12 library.
- currently same result as test hlavo-PR2
- can get sensor info
- can send measure request with responce
- does not receive any data (actually requests data in infinite loop)
- problems caused by randomly receiving null character (solved in `hlavo-station`)


## Teros31 sensors with Arduino

### Arduino_SDI12_Teros31
Minimalistic test. Not working.
Tests communication with Teros31 over SDI12 similar to PR2.

### Arduino_Serial_Teros31
Minimalistic test. Not working. Same as Arduino_SDI12_Teros31,
only using SoftwareSerial lib instead of SDI12.

### Arduino_bitwise_Teros31
Naive (failed) approach to read bitwise the DDI signal on power up.
