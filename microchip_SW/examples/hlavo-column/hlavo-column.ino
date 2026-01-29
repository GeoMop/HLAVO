
/*********************************************** COMMON ***********************************************/
#include <Every.h>
#include <Logger.h>
#include <math.h>

#define PIN_ON 47 // napajeni !!!

const char* setup_interrupt = "SETUP INTERRUPTED";

/************************************************ RUN ************************************************/
// Switch between testing/setup and long term run.
// #define TEST

#ifdef TEST
    /** TIMERS */
    // times in milliseconds, L*... timing level
    const int timer_L0_period = 1;      // [s] time reader
    const int timer_L1_period = 3;      // [s] read water height period
    const int timer_L2_period = 15;     // [s] date reading timer - PR2
    const int timer_L4_period = 10*60;  // [s] watchdog timer - 10 min
    #define VERBOSE 1
#else
    /** TIMERS */
    // times in milliseconds, L*... timing level
    const int timer_L0_period = 1;        // [s] time reader
    const int timer_L1_period = 1;        // [s] read water height period
    const int timer_L2_period = 5*60;     // [s] date reading timer - PR2
    const int timer_L4_period = 24*3600;  // [s] watchdog timer - 24 h
    #define VERBOSE 1
#endif

Every timer_L0(timer_L0_period*1000);     // time reader
Every timer_L1(timer_L1_period*1000);     // read water height timer
Every timer_L2(timer_L2_period*1000);     // date reading timer - PR2
Every timer_L4(timer_L4_period*1000);     // watchdog timer

// rain regimes timers
Timer timer_rain_wait(10*1000, false);      // wait between rain regimes, once timer
Timer timer_rain_regime(60*1000, false);    // rain regime, once timer

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
DateTime dt_now;

/************************************************ FLOW *************************************************/
#include "column_flow_data.h"
#include "linear_voltage_sensor.h"
#include "average_value.h"

#define VALVE_OUT_PIN 15

// ultrasonic sensor S18U
LinearVoltageSensor whs(5, 0.05, 3.13, 220, 30);  // pin, aVolt, bVolt, aVal, bVal

bool start_valve_out = false;
bool valve_out_open = false;

Timer timer_outflow(45*1000, false);    // time for pumping water out
const float outflow_min_wh = 200;              // [mm] limit water height to release
AverageValue outflow_wh(5);
float previous_height = 0;

char data_flow_filename[max_filepath_length] = "column_flow.csv";


/************************************************ RAIN *************************************************/
#include "rain_regime.h"

#define PUMP_IN_PIN 6
bool pump_in_finished = true;
bool rain_regime = true;
uint8_t current_rain_regime_idx = 0;
FileInfo current_rain_file("/current_rain.txt");
const bool force_fresh_rain_regimes = false;  // if true, ignore saved rain regime

LinearVoltageSensor rain_whs(7, 0.05, 2.21, 10, 630);  // pin, aVolt, bVolt, aVal, bVal
float rain_min_wh = 100; // mm
bool no_more_rain = false;
AverageValue rain_wh(5);

/**
  15s kropení každých 30min; po dobu 8 hodin, pak 4 hodiny čekat
  30s kropení každých 30min; po 4 hodiny, 8 hodin čekat
  60s kropení každých 30min; po 2 hodiny, 10 hod čekat
  120s kropení každých 30min; 1 hodina; 11 čekat
  15s kropení každých 15min; po dobu 8 hodin, pak 4 hodiny čekat
  30s kropení každých 15min; po 4 hodiny, 8 hodin čekat
  60s kropení každých 15min; po 2 hodiny, 10 hod čekat
  120s kropení každých 15min; 1 hodina; 11 čekat
 */
const int n_rain_regimes = 8;
// unsigned long tr_legnth_s, float tr_period_h, float length_h, float wait_h
RainRegime rain_regimes[n_rain_regimes] = {
  RainRegime(15, 0.5, 8.0, 4.0),
  RainRegime(30, 0.5, 4.0, 8.0),
  RainRegime(60, 0.5, 2.0, 10.0),
  RainRegime(120, 0.5, 1.0, 11.0),

  RainRegime(15, 0.25, 8.0, 4.0),
  RainRegime(30, 0.25, 4.0, 8.0),
  RainRegime(60, 0.25, 2.0, 10.0),
  RainRegime(120, 0.25, 1.0, 11.0),
};

// // DEBUG without peripheries and POWER
// const int n_rain_regimes = 4;
// RainRegime rain_regimes[n_rain_regimes] = {
//   RainRegime(3, 5./3600, 15./3600, 50./3600),
//   RainRegime(5, 8./3600, 16./3600, 5./3600),
//   RainRegime(4, 6./3600, 15./3600, 6./3600),
//   RainRegime(2, 4./3600, 24./3600, 5./3600)
// };

// DEBUG with POWER
// const int n_rain_regimes = 4;
// RainRegime rain_regimes[n_rain_regimes] = {
//   RainRegime(5, 36./3600, 108./3600, 9./3600),
//   RainRegime(8, 20./3600, 40./3600, 9./3600),
//   RainRegime(5, 18./3600, 108./3600, 9./3600),
//   RainRegime(8, 36./3600, 108./3600, 9./3600)
// };

void saveCurrentRain() {
  char text[100];
  DateTime dt = rain_regimes[current_rain_regime_idx].last_rain;
  snprintf(text, sizeof(text), "%s\n%d\n%d",
      dt.timestamp().c_str(),
      current_rain_regime_idx,
      rain_regimes[current_rain_regime_idx].counter);
  current_rain_file.write(text);
}

void loadCurrentRain() {

  Logger::printf(Logger::INFO, "loadCurrentrain.\n");

  if( ! SD.exists(current_rain_file.getPath()))
  {
      Logger::printf(Logger::INFO, "No last rain found.\n");
      current_rain_regime_idx = 0;
      return;
  }
  else
  {
    if(force_fresh_rain_regimes)
    {
        Logger::printf(Logger::INFO, "Removing last rain found.\n");
        SD.remove(current_rain_file.getPath());
        current_rain_regime_idx = 0;
        return;
    }
  }

  File file = SD.open(current_rain_file.getPath());
  if(!file){
      Logger::printf(Logger::ERROR, "Failed to open file for reading: %s\n", current_rain_file.getPath());
      current_rain_regime_idx = 0;
      return;
  }

  // Serial.print("Read from file: ");
  if(file.available()){
      // Logger::printf(Logger::INFO, "File available.\n");
      String dt_string = file.readStringUntil('\n');
      Logger::printf(Logger::INFO, "Read last rain dt: %s\n", dt_string.c_str());
      DateTime dt(dt_string.c_str());
      
      String idx_string = file.readStringUntil('\n');
      Logger::printf(Logger::INFO, "Read rain regime index: %s\n", idx_string.c_str());
      current_rain_regime_idx = idx_string.toInt();

      String counter_string = file.readStringUntil('\n');
      Logger::printf(Logger::INFO, "Read rain counter: %s\n", counter_string.c_str());
      uint8_t counter = counter_string.toInt() - 1;

      if(current_rain_regime_idx >= sizeof(rain_regimes))
      {
        Logger::printf(Logger::ERROR, "Invalid rain regime index: %d\n", current_rain_regime_idx);
        current_rain_regime_idx = 0;
        rain_regimes[current_rain_regime_idx].last_rain = dt;
        rain_regimes[current_rain_regime_idx].counter = 0;
      }
      else
      {
        char msg[100];
        Logger::printf(Logger::INFO, "Last rain regime [ORIG]:");
        hlavo::SerialPrintf(sizeof(msg), rain_regimes[current_rain_regime_idx].print(msg, sizeof(msg)));

        rain_regimes[current_rain_regime_idx].last_rain = dt;
        rain_regimes[current_rain_regime_idx].counter = counter;

        int rest_length = rain_regimes[current_rain_regime_idx].length
            - counter * rain_regimes[current_rain_regime_idx].trigger_period;
        if(rest_length>0)
          rain_regimes[current_rain_regime_idx].length = rest_length;
        else
        {
          // check end of regimes
          if(current_rain_regime_idx+1 >= n_rain_regimes)
          {
            Logger::printf(Logger::INFO, "All rain regimes finished.");
            no_more_rain = true;
            return;
          }
          else // check wait time
          {
            TimeSpan ts(rain_regimes[current_rain_regime_idx].wait / 1000);
            DateTime dt_wait_end = dt+ts;
            // dt+ts is end of wait time
            if( dt_wait_end > dt_now)
            {
              TimeSpan wait_ts = dt_wait_end - dt_now;
              Logger::printf(Logger::INFO, "WAIT REGIME for further %d s.", wait_ts.totalseconds());
              timer_rain_wait.reset(wait_ts.totalseconds()*1000);
              rain_regime = false;
              return;
            }
            // else continue with next regime
            current_rain_regime_idx++;
          }
        }

        Logger::printf(Logger::INFO, "Current rain regime [LOADED counter %d]:", counter);
        hlavo::SerialPrintf(sizeof(msg), rain_regimes[current_rain_regime_idx].print(msg, sizeof(msg)));
      }
  }
  else
  {
    Logger::printf(Logger::ERROR, "Failed to open file for reading: %s\n", current_rain_file.getPath());
    no_more_rain = true;
  }
  file.close();
}

void start_rain()
{
  // start rain
  digitalWrite(PUMP_IN_PIN, LOW);
  pump_in_finished = false;
  rain_regimes[current_rain_regime_idx].reset_timer_length(dt_now);
  Serial.printf("rain ON %d\n", rain_regimes[current_rain_regime_idx].counter);
  saveCurrentRain();
}

void stop_rain()
{
    // stop rain
  digitalWrite(PUMP_IN_PIN, HIGH);
  Serial.printf("rain OFF\n");
  pump_in_finished = true;
  // timer_rain_length.running = false;
}

void start_rain_regime()
{
  RainRegime& rain_reg = rain_regimes[current_rain_regime_idx];

  Serial.printf("RAIN REGIME: %d s\n", rain_reg.length/1000);
  char msg[100];
  hlavo::SerialPrintf(sizeof(msg), rain_reg.print(msg, sizeof(msg)));
  rain_regime = true;
  start_rain();
  timer_rain_regime.reset(rain_reg.length);
  rain_reg.reset_timer_period();
}

/******************************************* TEMP. AND HUM. *******************************************/
#include "Adafruit_SHT4x.h"
#include <Wire.h>  
#include "SparkFunBME280.h"
#include "bme280_data.h"
// https://www.laskakit.cz/senzor-tlaku--teploty-a-vlhkosti-bme280--1m/
// https://github.com/sparkfun/SparkFun_BME280_Arduino_Library/releases
// https://randomnerdtutorials.com/esp32-bme280-arduino-ide-pressure-temperature-humidity/
// set I2C address, default is 0x77, LaskaKit supplies with 0x76
const uint8_t tempSensor_I2C = 0x76;
BME280 tempSensor;
char data_bme280_filename[max_filepath_length] = "column_atmospheric.csv";

/********************************************** SDI12 COMM ********************************************/
#include "sdi12_comm.h"
#define SDI12_DATA_PIN 4         // The pin of the SDI-12 data bus

SDI12Comm sdi12_comm(SDI12_DATA_PIN, 1);  // (data_pin, verbose)

/********************************************* PR2 SENSORS ********************************************/
#include "pr2_data.h"
#include "pr2_reader.h"

const char pr2_address = '3';  // sensor addresses on SDI-12
PR2Reader pr2_reader = PR2Reader(&sdi12_comm, pr2_address);
char data_pr2_filename[max_filepath_length] = {"pr2_a3.csv"};

bool pr2_all_finished = false;


/********************************************* Teros31 SENSORS ********************************************/
#include "teros31_data.h"
#include "teros31_reader.h"
const uint8_t n_teros31_sensors = 3;
const char teros31_addresses[n_teros31_sensors] = {'A','B','C'};  // sensor addresses on SDI-12, 942, 947, 948

Teros31Reader teros31_readers[3] = {
  Teros31Reader(&sdi12_comm, teros31_addresses[0]),
  Teros31Reader(&sdi12_comm, teros31_addresses[1]),
  Teros31Reader(&sdi12_comm, teros31_addresses[2])
};
char data_teros31_filenames[n_teros31_sensors][max_filepath_length] = {"teros31_aA.csv", "teros31_aB.csv", "teros31_aC.csv"};
uint8_t teros31_iss = 0;  // current sensor reading
bool teros31_all_finished = false;


/****************************************** DATA COLLECTION ******************************************/


void measure_flow_values()
{
  // read water height
  float voltage;
  float height = whs.read(&voltage);
  Serial.printf("Voltage: %.2f    Height: %.2f\n", voltage, height);
  float rain_height = rain_whs.read(&voltage);
  Serial.printf("Voltage: %.2f    Rain Height: %.2f\n", voltage, rain_height);

  outflow_wh.add(height);
  rain_wh.add(rain_height);

  // check height limit, possibly run pump out
  if( (outflow_wh.average_unfilled() >= outflow_min_wh)
     || std::isnan(outflow_wh.average_unfilled())) // safe for overfilled column
  {
    start_valve_out = true; // open valve (once at a time)
    outflow_wh.reset();
    previous_height = std::numeric_limits<float>::quiet_NaN();
  }

  // check rain height limit, possibly stop raining
  if(rain_wh.average_unfilled() <= rain_min_wh)
  {
    stop_rain();
    no_more_rain = true;
    Serial.println("NO more water for raining.");
  }

  ColumnFlowData data;
  data.datetime = dt_now;
  data.pump_in = !pump_in_finished;   // is pump running?
  data.pump_out = valve_out_open;     // is valve open?
  data.rain_height = rain_wh.average();

  if(valve_out_open){
    // safe NaN during outflow
  }
  else if(!outflow_wh.isReady())
  {
    // safe NaN until window is filled
  }
  else
  {
    data.height = outflow_wh.average();
    // flux is difference of heights
    //TODO multiply by cross-section
    data.flux = (previous_height - data.height) / timer_L1_period;
    previous_height = data.height;
  }

  // write data
  CSVHandler::appendData(data_flow_filename, &data);

  // #ifdef TEST
  //   // TEST read data from CSV
  //   Serial.println("Flow data read");
  //   CSVHandler::printFile(data_flow_filename);
  // #endif
}

// use PR2 reader to request and read data from PR2
// minimize delays so that it does not block main loop
void collect_and_write_PR2()
{
  bool res = false;
  res = pr2_reader.TryRequest();
  if(!res)  // failed request
  {
    pr2_reader.Reset();
    return;
  }

  pr2_reader.TryRead();
  if(pr2_reader.finished)
  {
    pr2_reader.data.datetime = dt_now;
    // if(VERBOSE >= 1)
    {
      // Serial.printf("DateTime: %s. Writing PR2Data[a%d].\n", dt.timestamp().c_str(), pr2_address);
      char msg[400];
      hlavo::SerialPrintf(sizeof(msg)+20, "PR2[%c]: %s\n",pr2_address, pr2_reader.data.print(msg, sizeof(msg)));
    }

    // Logger::print("collect_and_write_PR2 - CSVHandler::appendData");
    CSVHandler::appendData(data_pr2_filename, &(pr2_reader.data));

    pr2_reader.Reset();
    pr2_all_finished = true;
  }
}

// use Teros31 reader to request and read data from Teros31
// minimize delays so that it does not block main loop
void collect_and_write_Teros31()
{
  const uint8_t sdelay = 100;
  const uint8_t stryouts = 20;

  if(!teros31_readers[teros31_iss].requested)
  {
    bool res = false;
    res = teros31_readers[teros31_iss].TryRequest();
    if(!res)  // failed request
    {
      if(teros31_readers[teros31_iss].n_tryouts <= stryouts)
      {
        delay(sdelay);
        return;
      }
      Logger::printf(Logger::ERROR, "Teros31 [a%c], request failed.", teros31_addresses[teros31_iss]);

      teros31_readers[teros31_iss].Reset();
      teros31_iss++;

      if(teros31_iss == n_teros31_sensors){
        teros31_iss = 0;
        teros31_all_finished = true;
      }
      return;
    }
  }

  teros31_readers[teros31_iss].TryRead();
  if(teros31_readers[teros31_iss].finished)
  {
    teros31_readers[teros31_iss].data.datetime = dt_now;
    // if(VERBOSE >= 1)
    {
      // Serial.printf("DateTime: %s. Writing PR2Data[a%d].\n", dt.timestamp().c_str(), pr2_addresses[teros31_iss]);
      char msg[400];
      hlavo::SerialPrintf(sizeof(msg)+20, "Teros31[%c]: %s\n",teros31_addresses[teros31_iss], teros31_readers[teros31_iss].data.print(msg, sizeof(msg)));
    }

    // Logger::print("collect_and_write_PR2 - CSVHandler::appendData");
    CSVHandler::appendData(data_teros31_filenames[teros31_iss], &(teros31_readers[teros31_iss].data));

    teros31_readers[teros31_iss].Reset();
    teros31_iss++;
    if(teros31_iss == n_teros31_sensors)
    {
      teros31_iss = 0;
      teros31_all_finished = true;
    }
  }
  else{
    if(teros31_readers[teros31_iss].n_tryouts <= stryouts)
    {
      delay(sdelay);
      return;
    }
    else{
      Logger::printf(Logger::ERROR, "Teros31 [a%c], read failed.", teros31_addresses[teros31_iss]);
      teros31_readers[teros31_iss].Reset();
      teros31_iss++;
      if(teros31_iss == n_teros31_sensors)
      {
        teros31_iss = 0;
        teros31_all_finished = true;
      }
    }
  }
}

void collect_and_write_atmospheric()
{
  // Serial.print("Humidity: ");
  // Serial.print(tempSensor.readFloatHumidity(), 0);
  // Serial.print(" Pressure: ");
  // Serial.print(tempSensor.readFloatPressure(), 0);
  // Serial.print(" Temp: ");
  // Serial.print(tempSensor.readTempC(), 2);
  // Serial.println();
  
  BME280_SensorMeasurements measurements;
  tempSensor.readAllMeasurements(&measurements); // tempScale = 0 => Celsius

  BME280Data data;
  data.datetime = dt_now;
  data.humidity = measurements.humidity;
  data.pressure = measurements.pressure;
  data.temperature = measurements.temperature;

  // write data
  CSVHandler::appendData(data_bme280_filename, &data);

  Serial.printf("Humidity: %.0f, Pressure: %.0f, Temperature: %.2f\n", measurements.humidity, measurements.pressure, measurements.temperature);
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

  // pump/valve pins, reset
  pinMode(VALVE_OUT_PIN, OUTPUT);
  digitalWrite(VALVE_OUT_PIN, HIGH);
  summary += " - VALVE OUT OFF (" +  String(VALVE_OUT_PIN) + " on)\n";
  pinMode(PUMP_IN_PIN, OUTPUT);
  digitalWrite(PUMP_IN_PIN, HIGH);
  summary += " - PUMP IN OFF (" +  String(PUMP_IN_PIN) + " off)\n";

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
  dt_now = rtc_clock.now();

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
  Logger::print("Log set up.");

  // Water Height sensor S18U
  whs.begin();

  rain_whs.setWindow(0.0,3.0);
  rain_whs.begin();

  // BME280 - temperature, pressure, humidity
  tempSensor.setI2CAddress(tempSensor_I2C); // set I2C address, default 0x77
  if(tempSensor.beginI2C())
  {
    tempSensor.setFilter(0); //0 to 4 is valid. Filter coefficient. See 3.4.4
    tempSensor.setStandbyTime(0); //0 to 7 valid. Time between readings. See table 27.
    tempSensor.setTempOverSample(1); //0 to 16 are valid. 0 disables temp sensing. See table 24.
    tempSensor.setPressureOverSample(1); //0 to 16 are valid. 0 disables pressure sensing. See table 23.
    tempSensor.setHumidityOverSample(1); //0 to 16 are valid. 0 disables humidity sensing. See table 19.
    tempSensor.setMode(MODE_FORCED); //MODE_SLEEP, MODE_FORCED, MODE_NORMAL is valid.
    summary += " - BME280 ready\n";
  }
  else
  {
    summary += " - BME280 FAILED\n";
    Logger::print("BME280 not found.", Logger::WARN);
  }

  // SDI12
  delay(1000);
  Serial.println("Opening SDI-12 for PR2...");
  sdi12_comm.begin();//

  delay(1000);  // allow things to settle
  // get info from all SDI12 sensors
  uint8_t nbytes = 0;
  String cmd = String(pr2_address) + "I!";
  // Logger::print(sdi12_comm.requestAndReadData(cmd.c_str(), &nbytes));  // Command to get sensor info
  char* msg = sdi12_comm.requestAndReadData(cmd.c_str(), &nbytes);
  delay(500);
  for(int i=0; i<n_teros31_sensors; i++){
    cmd = String(teros31_addresses[i]) + "I!";
    // Logger::print(sdi12_comm.requestAndReadData(cmd.c_str(), &nbytes));  // Command to get sensor info
    char* msg = sdi12_comm.requestAndReadData(cmd.c_str(), &nbytes);
    delay(500);
  }
  // while(1){delay(1000);}

  // Data files setup
  char csvLine[400];
  // flow data file
  const char* flow_dir="flow";
  CSVHandler::createFile(data_flow_filename,
                         ColumnFlowData::headerToCsvLine(csvLine, max_csvline_length),
                         dt_now, flow_dir);
  
  const char* atm_dir="atmospheric";
  CSVHandler::createFile(data_bme280_filename,
                         BME280Data::headerToCsvLine(csvLine, max_csvline_length),
                         dt_now, atm_dir);
  // PR2 data file
  const char* pr2_dir="pr2_sensor";
  CSVHandler::createFile(data_pr2_filename,
                         PR2Data::headerToCsvLine(csvLine, max_csvline_length),
                         dt_now, pr2_dir);
  // Teros31 data files
  for(int i=0; i<n_teros31_sensors; i++){
    char teros31_dir[20];
    sprintf(teros31_dir, "teros31_sensor_%d", i);
    CSVHandler::createFile(data_teros31_filenames[i],
                           Teros31Data::headerToCsvLine(csvLine, max_csvline_length),
                           dt_now, teros31_dir);
  }

  // for(int i=0; i<2; i++)
    // rain_regimes[i].compute_trigger();
  for(int i=0; i<n_rain_regimes; i++)
  {
    char msg[100];
    hlavo::SerialPrintf(sizeof(msg), rain_regimes[i].print(msg, sizeof(msg)));
  }

  loadCurrentRain();

  delay(5000);
  // Serial.flush();
  // while(1)
  //   ;

  delay(500);
  print_setup_summary(summary);
  // delay(5000);

  // synchronize timers after setup
  timer_L2.reset(true);
  timer_L1.reset(true);
  timer_L4.reset(false);

  // start first regime a start first rain
  if(! no_more_rain && rain_regime)
    start_rain_regime();

  // {
  //   // start rain
  //   digitalWrite(PUMP_IN_PIN, LOW);
  //   Serial.printf("rain ON\n");
  //   delay(5*60*1000);
  //   // stop rain
  //   digitalWrite(PUMP_IN_PIN, HIGH);
  //   Serial.printf("rain OFF\n");
  //   Serial.printf("rain counter %d of %d\n", rain_n_cycles, rain_max_n_cycles);
  // }

  // while(1);
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


// Runs every loop
// Checks the flag `start_valve_out` for output valve opening
// Checks the timer to close the output valve
void control_valve_out()
{
  if(start_valve_out)
  {
    digitalWrite(VALVE_OUT_PIN, LOW);
    timer_outflow.reset();
    valve_out_open = true;
    start_valve_out = false;
    Serial.printf("valve out ON\n");
  }

  if(timer_outflow.after() && valve_out_open)
  {
    digitalWrite(VALVE_OUT_PIN, HIGH);
    valve_out_open = false;
    Serial.printf("valve out OFF\n");
  }
}


void control_rain()
{
  if(no_more_rain)
  {
    return;
  }

  RainRegime& rain_reg = rain_regimes[current_rain_regime_idx];
  // rain interval timer
  if(rain_regime && rain_reg.timer_period())
  {
    start_rain();
    // Serial.printf("rain counter\n");
  }

  // rain timer finishes
  if(!pump_in_finished && rain_reg.timer_length())
  {
    stop_rain();
  }

  // rain regime timer finishes
  if(rain_regime && timer_rain_regime())
  {
    // stop_rain();
    rain_regime = false;
    timer_rain_wait.reset(rain_reg.wait);
    Serial.printf("WAIT REGIME: %d s\n", rain_reg.wait/1000);
  }

  // rain wait timer finishes
  if(timer_rain_wait())
  {
    current_rain_regime_idx++;
    if (current_rain_regime_idx >= n_rain_regimes)
    {
      Serial.println("RAIN REGIME all finished");
      return;
    }

    start_rain_regime();
  }
}

/*********************************************** LOOP ***********************************************/ 
void loop() {

  if(timer_L0()){
    dt_now = rtc_clock.now();
  }

  // if(timer_L1())
  // {
  //   measure_flow_values();
  // }
  // control_rain();
  // return; // skip the rest for testing rain
  

  if(timer_L1())
  {
    unsigned int L2_left = (timer_L2.interval + timer_L2.last - millis())/1000;
    Serial.printf("L1: %d s\n", L2_left);
    Serial.flush();
    measure_flow_values();

    if(!no_more_rain)
    {
      String s;
      if(rain_regime)
      {
        unsigned int t_left = (timer_rain_wait.interval + timer_rain_wait.last - millis())/1000;
        s = "R " + String(t_left) + " s";
      }
      else
      {
        unsigned int t_left = (timer_rain_wait.interval + timer_rain_wait.last - millis())/1000;
        s = "W " + String(t_left) + " s";
      }
      Serial.printf("RR[%d], c[%d] %s\n", current_rain_regime_idx, rain_regimes[current_rain_regime_idx].counter, s.c_str());
    }
  }
  control_valve_out();
  control_rain();

  // return; // skip the rest for testing rain

  // do not read data during rain (voltage source damages data signal)
  if(pump_in_finished)
  {
    // read values from PR2 and Teros31 sensors when reading not finished yet
    // and write to a file when last values received
    if(!pr2_all_finished)
      collect_and_write_PR2();
    else if(!teros31_all_finished){
      collect_and_write_Teros31();
    }

    if(timer_L2())
    {
      Serial.println("-------------------------- L2 TICK --------------------------");

      collect_and_write_atmospheric();

      if(teros31_all_finished && pr2_all_finished){
        pr2_all_finished = false;
        teros31_all_finished = false;
      }
    }
  }

  // // Serial.println("-------------------------- TICK --------------------------");
  // uint8_t nbytes = 0;
  // String cmd = String(pr2_address) + "I!";
  // // Logger::print(sdi12_comm.requestAndReadData(cmd.c_str(), &nbytes));  // Command to get sensor info
  // // delay(1000);
  // if(!pr2_all_finished)
  // {
  //   // sdi12_comm.commandTask(cmd.c_str());
  //   // if(!sdi12_comm.commandTaskRunning && !sdi12_comm.received)
  //   // {
  //   //   Serial.println("received");
  //   //   Serial.println(sdi12_comm.getData());
  //   //   pr2_all_finished = true;
  //   // }
  //   // Logger::print(sdi12_comm.requestAndReadData(cmd.c_str(), &nbytes));  // Command to get sensor info
  //   char* msg = sdi12_comm.requestAndReadData(cmd.c_str(), &nbytes);
  //   // Logger::printHex(msg, nbytes);
  //   // Logger::print(msg);
  //   // delay(300);
  //   pr2_all_finished = true;
  // }
  // else if(!teros31_all_finished)
  // {
  //     cmd = String(teros31_addresses[teros31_iss]) + "I!";
  //     // sdi12_comm.commandTask(cmd.c_str());
  //     // if(!sdi12_comm.commandTaskRunning && !sdi12_comm.received)
  //     // {
  //     //   Serial.println(sdi12_comm.getData());
  //     //   teros31_iss++;
  //     //   if(teros31_iss == n_teros31_sensors)
  //     //   {
  //     //     teros31_iss = 0;
  //     //     teros31_all_finished = true;
  //     //   }
  //     // }

  //     // Logger::print(sdi12_comm.requestAndReadData(cmd.c_str(), &nbytes));  // Command to get sensor info
  //     char* msg = sdi12_comm.requestAndReadData(cmd.c_str(), &nbytes);
  //     // Logger::printHex(msg, nbytes);
  //     // Logger::print(msg);
      
  //     teros31_iss++;
  //     if(teros31_iss == n_teros31_sensors)
  //     {
  //       teros31_iss = 0;
  //       teros31_all_finished = true;
  //     }
  //     // delay(300);

  //     // for(int i=0; i<n_teros31_sensors; i++){
  //     //   cmd = String(teros31_addresses[i]) + "I!";
  //     //   Logger::print(sdi12_comm.requestAndReadData(cmd.c_str(), &nbytes));  // Command to get sensor info
  //     //   // sdi12_comm.requestAndReadData(cmd.c_str(), &nbytes);
  //     //   delay(1000);
  //     // }
  // }

  if(timer_L4())
  {
    Serial.println("-------------------------- L4 TICK --------------------------");
    Logger::print("L4 TICK - Reboot");
    Serial.printf("\nReboot...\n\n");
    delay(250);
    ESP.restart();
  }
}