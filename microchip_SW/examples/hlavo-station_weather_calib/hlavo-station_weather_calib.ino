#include "meteo_data.h"

#define PIN_ON 47 // napajeni !!!

/*********************************************** SD CARD ***********************************************/
// SD card IO
#include "SD.h"
// file handling
#include "file_info.h"
// SD card pin
#define SD_CS_PIN 10
#define data_meteo_filename "/meteo.txt"


/************************************************* RTC *************************************************/
// definice sbernice i2C pro RTC (real time clock)
#define rtc_SDA_PIN 42 // data pin
#define rtc_SCL_PIN 2  // clock pin
#include "clock.h"
Clock rtc_clock(rtc_SDA_PIN, rtc_SCL_PIN);

DateTime dt_start;

/*********************************************** BATTERY ***********************************************/
#include "ESP32AnalogRead.h"
ESP32AnalogRead adc;
#define ADCpin 9
#define DeviderRatio 1.7693877551  // Voltage devider ratio on ADC pin 1MOhm + 1.3MOhm


/******************************************* TEMP. AND HUM. *******************************************/
#include "Adafruit_SHT4x.h"
Adafruit_SHT4x sht4 = Adafruit_SHT4x();

#include "BH1750.h"
BH1750 lightMeter;

/****************************************** WHEATHER STATION ******************************************/
#define WEATHER_PERIOD 2  // data refresh in seconds
#define WINDVANE_PIN A0   // A0 := 1
#define ANEMOMETER_PIN 5
#define RAINGAUGE_PIN 6  // 10 kOhm / 10pF
#include "weather_station.h"
WeatherStation weather(WINDVANE_PIN, ANEMOMETER_PIN, RAINGAUGE_PIN, WEATHER_PERIOD);

// interuption
// ICACHE_RAM_ATTR replaced by IRAM_ATTR (esp and arduino>3.0.0)
void IRAM_ATTR intAnemometer() { weather.intAnemometer(); }
void IRAM_ATTR intRaingauge() { weather.intRaingauge(); }
void IRAM_ATTR intPeriod() { weather.intTimer(); }


/*********************************************** SETUP ***********************************************/ 
void setup() {
  Serial.begin(115200);
  while (!Serial)
  {
      ; // cekani na Serial port
  }

  Serial.println("HLAVO project starts.");

  // for version over 3.5 need to turn uSUP ON
  Serial.print("set power pin: "); Serial.println(PIN_ON);
  pinMode(PIN_ON, OUTPUT);      // Set EN pin for uSUP stabilisator as output
  digitalWrite(PIN_ON, HIGH);   // Turn on the uSUP power

  // clock setup
  rtc_clock.begin();

  // weather station
  weather.setup(intAnemometer, intRaingauge, intPeriod);

  // battery
  adc.attach(ADCpin); // setting ADC

  // humidity and temperature
  if (! sht4.begin())
  {
    Serial.println("SHT4x not found.");
  }

  sht4.setPrecision(SHT4X_HIGH_PRECISION); // nejvyssi rozliseni
  sht4.setHeater(SHT4X_NO_HEATER); // bez vnitrniho ohrevu

  // Light
  if(!lightMeter.begin())
  {
    Serial.println("BH1750 (light) not found.");
  }

  // SD card setup
  pinMode(SD_CS_PIN, OUTPUT);
  // SD Card Initialization
  if (SD.begin()){
      Serial.println("SD card is ready to use.");
  }
  else{
      Serial.println("SD card initialization failed");
      return;
  }

  // char csvLine[150];
  // FileInfo datafile(SD, data_meteo_filename);
  // datafile.remove();
  // // datafile.read();
  // if(!datafile.exists())
  //   datafile.write(MeteoData::headerToCsvLine(csvLine));

  dt_start = rtc_clock.now();
  Serial.println("setup completed.");
  Serial.println("--------------------------");
}


unsigned int speed_ticks = 0;
unsigned int rain_ticks = 0;

/*********************************************** LOOP ***********************************************/ 
void loop() {
  
  weather.update();

	if (weather.gotData()) {

    // sensors_event_t humidity, temp; // promenne vlhkost a teplota
    // sht4.getEvent(&humidity, &temp);
    // float light_lux = lightMeter.readLightLevel();

    DateTime dt = rtc_clock.now();
    TimeSpan dt_span = dt - dt_start;
    
    // data.datetime = dt;
    // data.wind_direction = weather.getDirection();
    speed_ticks += weather.getSpeedTicks();
    rain_ticks += weather.getRainTicks();

    // data.temperature = temp.temperature;
    // data.humidity = humidity.temperature;
    // data.light = light_lux;

    // data.battery_voltage = adc.readVoltage() * DeviderRatio;

    // Serial.printf("DateTime: %s\n", dt.timestamp().c_str());
    Serial.printf("TimeSpan: %d\n", dt_span.totalseconds());
  
    // Serial.printf("Temperature: %f degC\n", temp.temperature);
    // Serial.printf("Humidity: %f rH\n", humidity.relative_humidity);
    // Serial.printf("Light: %f lx\n", light_lux);

    // Serial.printf("Wind direc adc:  %d\n", weather.getDirAdcValue());
    // Serial.printf("Wind direc deg:  %f\n", data.wind_direction);
    Serial.printf("Wind speed TICK: %d\n", speed_ticks);
    Serial.printf("Wind speed TICK: %d\n", speed_ticks);
    Serial.printf("Wind speed [m/s]: %.2f\n", weather.getSpeed());
    Serial.printf("Rain gauge TICK: %d\n", rain_ticks);
    Serial.printf("Rain gauge [ml/min]: %.2f\n", weather.getRain_ml());
    // Serial.printf("Battery [V]: %f\n", data.battery_voltage);

    // char csvLine[150];
    // FileInfo datafile(SD, data_meteo_filename);
    // datafile.append(data.dataToCsvLine(csvLine));

    // datafile.read();

    weather.resetGotData();
    Serial.println("--------------------------");
  }

  // delay(1);
}