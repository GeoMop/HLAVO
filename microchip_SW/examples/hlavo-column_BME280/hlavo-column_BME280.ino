
/*********************************************** COMMON ***********************************************/
#include <Every.h>
#include <Logger.h>

#define PIN_ON 47 // napajeni !!!

const char* setup_interrupt = "SETUP INTERRUPTED";

/************************************************ RUN ************************************************/
/** TIMERS */
// times in milliseconds, L*... timing level
Every timer_L1(2000);
#define VERBOSE 1



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


/******************************************* TEMP. AND HUM. *******************************************/
#include "Adafruit_SHT4x.h"
#include <Wire.h>  
#include "SparkFunBME280.h"
// https://www.laskakit.cz/senzor-tlaku--teploty-a-vlhkosti-bme280--1m/
// https://github.com/sparkfun/SparkFun_BME280_Arduino_Library/releases
// https://randomnerdtutorials.com/esp32-bme280-arduino-ide-pressure-temperature-humidity/
// set I2C address, default is 0x77, LaskaKit supplies with 0x76
const uint8_t tempSensor_I2C = 0x76;
BME280 tempSensor;


#include "BH1750.h"
// default I2C address 0x23 (set in constructor)
BH1750 lightMeter;

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
  if (SD.begin()){
      Serial.println("SD card is ready to use.");
      summary += " - SD card [pin " + String(SD_CS_PIN) + "] ready \n";
  }
  else{
      Serial.println("SD card initialization failed.");
      Serial.println(setup_interrupt);
      while(1){delay(1000);}
  }
  Logger::setup_log(rtc_clock, "logs");
  Serial.println("Log set up.");
  Logger::print("Log set up.");

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

  // BME280 - temperature, pressure, humidity
  tempSensor.setI2CAddress(tempSensor_I2C); // set I2C address, default 0x77
  if(tempSensor.beginI2C())
  {
    summary += " - BME280 ready\n";
  }
  else
  {
    summary += " - BME280 FAILED\n";
    Logger::print("BME280 not found.", Logger::WARN);
  }
  
  print_setup_summary(summary);
  // scan_I2C();
  delay(5000);
  // while(1){delay(1000);}

  // synchronize timers after setup
  timer_L1.reset(true);
}

void print_setup_summary(String summary)
{
  summary = "\nSETUP SUMMARY:\n" + summary;
  summary = "\n=======================================================================\n" + summary + "\n";
  summary += F("INO file: " __FILE__ " " __DATE__ " " __TIME__ "\n\n");
  summary += "=======================================================================";

  Logger::print(summary);
  Logger::print("HLAVO station is running");
}


void scan_I2C()
{
  byte error, address;
  int nDevices;
  Serial.println("Scanning...");
  nDevices = 0;
  for(address = 1; address < 127; address++ ) {
    Wire.beginTransmission(address);
    error = Wire.endTransmission();
    if (error == 0) {
      Serial.print("I2C device found at address 0x");
      if (address<16) {
        Serial.print("0");
      }
      Serial.println(address,HEX);
      nDevices++;
    }
    else if (error==4) {
      Serial.print("Unknow error at address 0x");
      if (address<16) {
        Serial.print("0");
      }
      Serial.println(address,HEX);
    }    
  }
  if (nDevices == 0) {
    Serial.println("No I2C devices found\n");
  }
  else {
    Serial.println("done\n");
  }
  delay(5000);
}




/*********************************************** LOOP ***********************************************/ 
void loop() {
  
  // read values to buffer at fine time scale [fine Meteo Data]
  if(timer_L1())
  {
    Serial.println("        -------------------------- L1 TICK --------------------------");

    Serial.println(rtc_clock.now().timestamp().c_str());

    float light_lux = lightMeter.readLightLevel();
    Serial.print("Light: ");
    Serial.println(light_lux);

    Serial.print("Humidity: ");
    Serial.print(tempSensor.readFloatHumidity(), 0);

    Serial.print(" Pressure: ");
    Serial.print(tempSensor.readFloatPressure(), 0);

    Serial.print(" Temp: ");
    Serial.print(tempSensor.readTempC(), 2);

    Serial.println();

  }

  // scan_I2C();
}