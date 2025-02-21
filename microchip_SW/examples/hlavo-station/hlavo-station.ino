
/*********************************************** COMMON ***********************************************/
#include <Every.h>
#include <Logger.h>

#define PIN_ON 47 // napajeni !!!

const char* setup_interrupt = "SETUP INTERRUPTED";

/************************************************ RUN ************************************************/
// Switch between testing/setup and long term run.
// #define TEST
// PR2 - a1 - s Oddyssey U04 pod stromy
// PR2 - a0 - u meteo stanice

#ifdef TEST
    /** TIMERS */
    // times in milliseconds, L*... timing level
    Every timer_L1(1000);      // fine timer - humidity, temperature, meteo, ...
    // L2 - hardware timer with L2_WEATHER_PERIOD in seconds
    #define L2_WEATHER_PERIOD 10
    Every timer_L3(40*1000); // coarse timer - PR2 - 40 s
    Every timer_L4(10*60*1000);  // watchdog timer - 10 min
    #define VERBOSE 1
#else
    /** TIMERS */
    // times in milliseconds, L*... timing level
    Every timer_L1(1000);         // fine timer - humidity, temperature, ...
    // L2 - hardware timer with L2_WEATHER_PERIOD in seconds
    #define L2_WEATHER_PERIOD 60
    Every timer_L3(900*1000);     // coarse timer - PR2 - 15 min
    Every timer_L4(24*3600*1000); // watchdog timer - 24 h
    #define VERBOSE 1
#endif


/*********************************************** SD CARD ***********************************************/
// SD card IO
#include "CSV_handler.h"
// SD card pin
#define SD_CS_PIN 10


/************************************************* I2C *************************************************/
#include <Wire.h>
#define I2C_SDA_PIN 42 // data pin
#define I2C_SCL_PIN 2  // clock pin

/************************************************* RTC *************************************************/
// definice sbernice i2C pro RTC (real time clock)
// I2C address 0x68
#include "clock.h"
Clock rtc_clock;


/*********************************************** BATTERY ***********************************************/
#include "ESP32AnalogRead.h"
ESP32AnalogRead adc;
#define ADCpin 9
#define DeviderRatio 1.7693877551  // Voltage devider ratio on ADC pin 1MOhm + 1.3MOhm


/******************************************* TEMP. AND HUM. *******************************************/
#include "Adafruit_SHT4x.h"
Adafruit_SHT4x sht4 = Adafruit_SHT4x();

#include "BH1750.h"
// default I2C address 0x23 (set in constructor)
BH1750 lightMeter;

/****************************************** WHEATHER STATION ******************************************/
#define WINDVANE_PIN A0   // A0 := 1
#define ANEMOMETER_PIN 5
#define RAINGAUGE_PIN 6  // 10 kOhm, pullup
#include "weather_station.h"
#include "meteo_data.h"
char data_meteo_filename[max_filepath_length] = "meteo.csv";
WeatherStation weather(WINDVANE_PIN, ANEMOMETER_PIN, RAINGAUGE_PIN, L2_WEATHER_PERIOD);

// interuption
// ICACHE_RAM_ATTR replaced by IRAM_ATTR (esp and arduino>3.0.0)
void IRAM_ATTR intAnemometer() { weather.intAnemometer(); }
void IRAM_ATTR intRaingauge() { weather.intRaingauge(); }
void IRAM_ATTR intPeriod() { weather.intTimer(); }

/******************************************** PR2 SENSORS ********************************************/
#include "sdi12_comm.h"
#include "pr2_data.h"
#include "pr2_reader.h"

#define PR2_POWER_PIN 7        // The pin PR2 power
#define PR2_DATA_PIN 4         // The pin of the SDI-12 data bus
SDI12Comm sdi12_comm(PR2_DATA_PIN, 1);  // (data_pin, verbose)
const uint8_t n_pr2_sensors = 2;
const char pr2_addresses[n_pr2_sensors] = {'0','1'};  // sensor addresses on SDI-12
PR2Reader pr2_readers[2] = {        // readers enable reading all sensors without blocking loop
  PR2Reader(&sdi12_comm, pr2_addresses[0]),
  PR2Reader(&sdi12_comm, pr2_addresses[1])
};
char data_pr2_filenames[n_pr2_sensors][max_filepath_length] = {"pr2_a0.csv", "pr2_a1.csv"};

uint8_t iss = 0;  // current sensor reading
bool pr2_all_finished = false;

Timer timer_PR2_power(2000, false);

/****************************************** DATA COLLECTION ******************************************/
// L1 timer data buffer
float fineDataBuffer[NUM_FINE_VALUES][FINE_DATA_BUFSIZE];
int num_fine_data_collected = 0;

// L3 timer data buffer
MeteoData meteoDataBuffer[METEO_DATA_BUFSIZE];
int num_meteo_data_collected = 0;

// collect meteo data at fine interval into a fine buffer of floats
void fine_data_collect()
{
  sensors_event_t humidity, temp;
  bool sht4_res = sht4.getEvent(&humidity, &temp);
  float light_lux = lightMeter.readLightLevel();
  float battery = adc.readVoltage() * DeviderRatio;

  // should not happen
  if(num_fine_data_collected >= FINE_DATA_BUFSIZE)
  {
    Logger::printf(Logger::ERROR, "Warning: Reached maximal buffer size for fine timer (%d)\n", num_fine_data_collected);
    meteo_data_collect();
    num_fine_data_collected = 0;
  }

  int i = num_fine_data_collected;
  if(sht4_res){
    fineDataBuffer[0][i] = humidity.relative_humidity;
    fineDataBuffer[1][i] = temp.temperature;
  }
  else{
    fineDataBuffer[0][i] = 0.0f;
    fineDataBuffer[1][i] = -0.0f;
  }

  fineDataBuffer[2][i] = light_lux;
  fineDataBuffer[3][i] = battery;

  if(VERBOSE >= 1)
  {
    Serial.printf("        %d:  Hum. %.2f, Temp. %.2f, Light %.0f, Bat. %.3f\n", num_fine_data_collected,
      fineDataBuffer[0][i], fineDataBuffer[1][i], fineDataBuffer[2][i], fineDataBuffer[3][i]);
  }

  num_fine_data_collected++;
}

// compute statistics over the fine meteo data
// and save the meteo data into buffer of MeteoData
void meteo_data_collect()
{
  DateTime dt = rtc_clock.now();
  // Serial.printf("    DateTime: %s. Buffering MeteoData.\n", dt.timestamp().c_str());

  // should not happen
  if(num_meteo_data_collected >= METEO_DATA_BUFSIZE)
  {
    Logger::printf(Logger::ERROR, "Warning: Reached maximal buffer size for meteo data (%d)", num_meteo_data_collected);
    meteo_data_write();
    num_meteo_data_collected = 0;
  }

  MeteoData &data = meteoDataBuffer[num_meteo_data_collected];
  data.datetime = dt;
  data.compute_statistics(fineDataBuffer, NUM_FINE_VALUES, num_fine_data_collected);

  if (weather.gotData())
  {
    data.wind_direction = weather.getDirection();
    data.wind_speed = weather.getSpeed();
    data.raingauge = weather.getRain_ml();
  }

  if(VERBOSE >= 1)
  {
    char msg[400];
    Serial.printf("    %s\n", data.print(msg, sizeof(msg)));
    // data.dataToCsvLine(msg);
    // Serial.println(msg);
  }

  // write data into buffer
  num_meteo_data_collected++;

  // start over from the beginning of buffer
  num_fine_data_collected = 0;
}

// write the meteo data buffer to CSV
void meteo_data_write()
{
  // Fill the base class pointer array with addresses of derived class objects
  Logger::printf(Logger::INFO, "meteo_data_write: %d collected\n", num_meteo_data_collected);
  if(num_meteo_data_collected < 1)
    return;

  DataBase* dbPtr[num_meteo_data_collected];
  for (int i = 0; i < num_meteo_data_collected; i++) {
      dbPtr[i] = &meteoDataBuffer[i];
  }

  Logger::print("meteo_data_write - CSVHandler::appendData");
  CSVHandler::appendData(data_meteo_filename, dbPtr, num_meteo_data_collected);
  // start over from the beginning of buffer
  num_meteo_data_collected = 0;
}

// use PR2 reader to request and read data from PR2
// minimize delays so that it does not block main loop
void collect_and_write_PR2()
{
  bool res = false;
  res = pr2_readers[iss].TryRequest();
  if(!res)  // failed request
  {
    Serial.printf("TryRequest FAILED from PR2 address: %c\n", pr2_addresses[iss]);
    pr2_readers[iss].Reset();
    iss++;

    if(iss >= n_pr2_sensors)
      iss = 0;
    return;
  }

  pr2_readers[iss].TryRead();
  if(pr2_readers[iss].finished)
  {
    DateTime dt = rtc_clock.now();
    pr2_readers[iss].data.datetime = dt;
    // if(VERBOSE >= 1)
    {
      char msg[400];
      hlavo::SerialPrintf(sizeof(msg)+20, "PR2[%c]: %s\n",pr2_addresses[iss], pr2_readers[iss].data.print(msg, sizeof(msg)));
    }

    // Logger::print("collect_and_write_PR2 - CSVHandler::appendData");
    CSVHandler::appendData(data_pr2_filenames[iss], &(pr2_readers[iss].data));

    pr2_readers[iss].Reset();
    iss++;
    if(iss == n_pr2_sensors)
    {
      iss = 0;
      pr2_all_finished = true;
      set_power_for_pr2(false);
    }
  }
}

void set_power_for_pr2(bool on)
{
  if(on)
    setPin(PR2_POWER_PIN, LOW);  // turn on power for PR2
  else
    setPin(PR2_POWER_PIN, HIGH); // turn off power for PR2

}

/*********************************************** SETUP ***********************************************/
void setup() {
  Serial.begin(115200);
  while (!Serial)
  {
      ; // cekani na Serial port
  }
  String summary = "";

  Serial.println("Starting HLAVO station setup.");

  // necessary for I2C
  // for version over 3.5 need to turn uSUP ON
  Serial.print("set power pin: "); Serial.println(PIN_ON);
  pinMode(PIN_ON, OUTPUT);      // Set EN pin for uSUP stabilisator as output
  digitalWrite(PIN_ON, HIGH);   // Turn on the uSUP power
  summary += " - POWER PIN " +  String(PIN_ON) + " on\n";


  // I2C setup
  if(Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN))
  {
    Serial.println("TwoWire (I2C) is ready to use.");
    summary += " - I2C [SDA " + String(I2C_SDA_PIN) + " SCL " + String(I2C_SCL_PIN) + "] ready\n";
  }
  else
  {
    Serial.println("TwoWire (I2C) initialization failed.");
    Serial.println(setup_interrupt);
    while(1){delay(1000);}
  }

  // clock setup
  if(rtc_clock.begin())
  {
    Serial.println("RTC is ready to use.");
    summary += " - RTC ready\n";
  }
  else
  {
    Serial.println("RTC initialization failed.");
    Serial.println(setup_interrupt);
    while(1){delay(1000);}
  }
  DateTime dt = rtc_clock.now();

  // SD card setup
  pinMode(SD_CS_PIN, OUTPUT);
  // SD Card Initialization
  if (SD.begin(SD_CS_PIN)){
      Serial.println("SD card is ready to use.");
      summary += " - SD card [pin " + String(SD_CS_PIN) + "] ready \n";
  }
  else{
      Serial.println("SD card initialization failed.");
      Serial.println(setup_interrupt);
      while(1){delay(1000);}
  }
  // Logger::setup_log(rtc_clock, "logs");
  // Serial.println("Log set up.");
  Logger::print("Log set up.");


  // weather station
  weather.setup(intAnemometer, intRaingauge, intPeriod);

  // battery
  adc.attach(ADCpin); // setting ADC

  // humidity and temperature
  if (sht4.begin())
  {
    summary += " - SHT4x ready\n";
  }
  else
  {
    summary += " - SHT4x FAILED\n";
    Logger::print("SHT4x not found.", Logger::WARN);
  }

  sht4.setPrecision(SHT4X_HIGH_PRECISION); // nejvyssi rozliseni
  sht4.setHeater(SHT4X_NO_HEATER); // bez vnitrniho ohrevu

  // BH1750 - Light
  if(lightMeter.begin())
  {
    summary += " - BH1750 ready\n";
  }
  else
  {
    summary += " - BH1750 FAILED\n";
    Logger::print("BH1750 (light) not found.", Logger::WARN);
  }

  // Data files setup
  char csvLine[max_csvline_length];
  const char* meteo_dir="meteo";
  CSVHandler::createFile(data_meteo_filename,
                            MeteoData::headerToCsvLine(csvLine, max_csvline_length),
                            dt, meteo_dir);
  for(int i=0; i<n_pr2_sensors; i++){
    char pr2_dir[20];
    sprintf(pr2_dir, "pr2_sensor_%d", i);
    CSVHandler::createFile(data_pr2_filenames[i],
                              PR2Data::headerToCsvLine(csvLine, max_csvline_length),
                              dt, pr2_dir);
  }

  print_setup_summary(summary);
  delay(5000);

  // PR2
  pinMode(PR2_POWER_PIN, OUTPUT);
  set_power_for_pr2(true);
  timer_PR2_power.reset();

  {
    Serial.print("APPEND\n");
    FileInfo datafile(data_pr2_filenames[1]);
    datafile.append("FOO\n");
  }

  Serial.println("Opening SDI-12 for PR2...");
  sdi12_comm.begin();

  delay(1000);  // allow things to settle

  uint8_t nbytes = 0;
  for(int i=0; i<n_pr2_sensors; i++){
    String cmd = String(pr2_addresses[i]) + "I!";
    sdi12_comm.requestAndReadData(cmd.c_str(), &nbytes);  // Command to get sensor info
    delay(500);
  }

  // while(1){delay(100);}
  delay(1000);

  // synchronize timers after setup
  timer_L3.reset(true);
  timer_L1.reset(true);
  timer_L4.reset(false);
}

void print_setup_summary(String summary)
{
  summary = "\nSETUP SUMMARY:\n" + summary;
  summary = "\n=======================================================================\n" + summary + "\n";
  summary += F("INO file: " __FILE__ " " __DATE__ " " __TIME__ "\n\n");
  summary += "=======================================================================";

  Serial.print(summary); Serial.println("");
  // Logger::print(summary);
  Logger::print("HLAVO station is running");
}

/*********************************************** LOOP ***********************************************/ 
void loop() {
  
  
  weather.update();

  // read values to buffer at fine time scale [fine Meteo Data]
  if(timer_L1())
  {
    Serial.printf("        -------------------------- L1 TICK -------------------------- till L3: %d s\n", (timer_L3.interval + timer_L3.last - millis())/1000);
    fine_data_collect();
    FileInfo datafile(data_pr2_filenames[1]);
    datafile.append("HOO\n");
  }

  // read values to buffer at fine time scale [averaged Meteo Data]
	if (weather.gotData()) {
    Serial.println("    **************************************** L2 TICK ****************************************");

    meteo_data_collect();
    weather.resetGotData();
    Serial.println("    **************************************** ******* ****************************************");
  }

  
  // read values from PR2 sensors when reading not finished yet
  // and write to a file when last values received
  if(!pr2_all_finished && timer_PR2_power.after())
    collect_and_write_PR2();

  // request reading from PR2 sensors
  // and write Meteo Data buffer to a file
  if(timer_L3())
  {
    Serial.println("-------------------------- L3 TICK --------------------------");
    Logger::print("L3 TICK");
    meteo_data_write();

    pr2_all_finished = false;
    // Serial.println("PR2 power on.");
    set_power_for_pr2(true);
    timer_PR2_power.reset();

    #ifdef TEST
      // TEST read data from CSV
      // CSVHandler::printFile(data_meteo_filename);
      // for(int i=0; i<n_pr2_sensors; i++){
      //   CSVHandler::printFile(data_pr2_filenames[i]);
      // }
    #endif
  }

  if(timer_L4())
  {
    Serial.println("-------------------------- L4 TICK --------------------------");
    Logger::print("L4 TICK - Reboot");
    Serial.printf("\nReboot...\n\n");
    delay(250);
    ESP.restart();
  }
}