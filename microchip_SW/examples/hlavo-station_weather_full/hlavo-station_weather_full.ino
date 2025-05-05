#include "meteo_data.h"
#include <Every.h>

#define PIN_ON 47 // napajeni !!!

/** TIMERS */
// times in milliseconds, L*... timing level
Every timer_L1(1000); // fine timer - humidity, temperature, ...
// L2 - hardware timer with WEATHER_PERIOD
Every timer_L3(30000); // coarse timer - PR2 - TEST 30 s
// Every timer_L3(900000); // coarse timer - PR2 - RUN 15 min

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
// TODO: 60 s
#define WEATHER_PERIOD 10  // data refresh in seconds (TEST 10 s, RUN 60 s)
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


/****************************************** DATA COLLECTION ******************************************/
// L1 timer data buffer
float fineDataBuffer[NUM_FINE_VALUES][FINE_DATA_BUFSIZE];
int num_fine_data_collected = 0;

// L3 timer data buffer
MeteoData meteoDataBuffer[METEO_DATA_BUFSIZE];
int num_meteo_data_collected = 0;

// collects data at fine interval
void fine_data_collect()
{
  sensors_event_t humidity, temp;
  sht4.getEvent(&humidity, &temp);
  float light_lux = lightMeter.readLightLevel();
  float battery = adc.readVoltage() * DeviderRatio;

  // should not happen
  if(num_fine_data_collected >= FINE_DATA_BUFSIZE)
  {
    Serial.printf("Warning: Reached maximal buffer size for fine timer (%d)\n", num_fine_data_collected);
    meteo_data_collect();
    num_fine_data_collected = 0;
  }

  int i = num_fine_data_collected;
  fineDataBuffer[0][i] = humidity.relative_humidity;
  fineDataBuffer[1][i] = temp.temperature;
  fineDataBuffer[2][i] = light_lux;
  fineDataBuffer[3][i] = battery;

  Serial.printf("%d:   %.2f,  %.2f,  %.0f,  %.3f\n", num_fine_data_collected,
    fineDataBuffer[0][i], fineDataBuffer[1][i], fineDataBuffer[2][i], fineDataBuffer[3][i]);

  num_fine_data_collected++;
}

// save the meteo data to buffer
void meteo_data_collect()
{
  DateTime dt = rtc_clock.now();
  Serial.printf("DateTime: %s. Writing MeteoData.\n", dt.timestamp().c_str());

  // should not happen
  if(num_meteo_data_collected >= METEO_DATA_BUFSIZE)
  {
    Serial.printf("Warning: Reached maximal buffer size for meteo data (%d)\n", num_meteo_data_collected);
    all_data_write();
    num_meteo_data_collected = 0;
  }

  // MeteoData &data = meteoDataBuffer[num_meteo_data_collected];
  MeteoData data;
  data.datetime = dt;
  data.compute_statistics(fineDataBuffer, NUM_FINE_VALUES, num_fine_data_collected);

  if (weather.gotData())
  {
    data.wind_direction = weather.getDirection();
    data.wind_speed = weather.getSpeed();
    data.raingauge = weather.getRain_mm();
    Serial.printf("%d:   %.2f,  %d  %.2f,  %d  %.2f\n", num_meteo_data_collected,
    data.wind_direction, weather.getSpeedTicks(), data.wind_speed, weather.getRainTicks(), data.raingauge);
  }

  // write data into buffer
  meteoDataBuffer[num_meteo_data_collected] = data;
  num_meteo_data_collected++;

  // start over from the beginning of buffer
  num_fine_data_collected = 0;
}

// save the meteo data buffer to CSV
void all_data_write()
{
  // FileInfo datafile(SD, data_meteo_filename);
  char csvLine[100];

  File file = SD.open(data_meteo_filename, FILE_APPEND);
  if(!file){
      Serial.println("Failed to open file for appending");
  }
  else
  {
    for(int i=0; i<num_meteo_data_collected; i++)
    {
      bool res = file.print(meteoDataBuffer[i].dataToCsvLine(csvLine));
      if(!res){
          Serial.println("Append failed");
      }
    }
  }
  file.close();

  // start over from the beginning of buffer
  num_meteo_data_collected = 0;
}

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

  char csvLine[200];
  FileInfo datafile(SD, data_meteo_filename);
  datafile.remove();
  // datafile.read();
  if(!datafile.exists())
    datafile.write(MeteoData::headerToCsvLine(csvLine));

  Serial.println("setup completed.");
  Serial.println(F("Start loop " __FILE__ " " __DATE__ " " __TIME__));
  Serial.println("--------------------------");

  // synchronize timers after setup
  timer_L3.reset(true);
  timer_L1.reset(true);
}


/*********************************************** LOOP ***********************************************/ 
void loop() {
  
  weather.update();

  // read value to buffer at fine time scale
  if(timer_L1())
  {
    // Serial.printf("L1 tick\n");
    fine_data_collect();
  }

	if (weather.gotData()) {
    // Serial.printf("L2 tick - Weather\n");

    // sensors_event_t humidity, temp; // promenne vlhkost a teplota
    // sht4.getEvent(&humidity, &temp);
    // float light_lux = lightMeter.readLightLevel();

    // DateTime dt = rtc_clock.now();

    // Serial.printf("DateTime: %s\n", dt.timestamp().c_str());
    // Serial.printf("Temperature: %f degC\n", temp.temperature);
    // Serial.printf("Humidity: %f rH\n", humidity.relative_humidity);
    // Serial.printf("Light: %f lx\n", light_lux);

    // Serial.printf("Wind direc adc:  %d\n", weather.getDirAdcValue());
    // Serial.printf("Wind direc deg:  %f\n", weather.getDirection());
    // Serial.printf("Wind speed TICK: %d\n", weather.getSpeedTicks());
    // Serial.printf("Rain gauge TICK: %d\n", weather.getRainTicks());
    // Serial.printf("Battery [V]: %f\n", adc.readVoltage() * DeviderRatio);

    meteo_data_collect();

    weather.resetGotData();
    Serial.println("--------------------------");
  }

  // read value to buffer at fine time scale
  if(timer_L3())
  {
    // Serial.printf("L3 tick\n");
    all_data_write();

    // TEST read data from CSV
    FileInfo datafile(SD, data_meteo_filename);
    datafile.read();
  }
}