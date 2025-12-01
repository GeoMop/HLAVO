
/*********************************************** COMMON ***********************************************/
#include <Every.h>
#include <Logger.h>

#define PIN_ON 47 // napajeni !!!

const char* setup_interrupt = "SETUP INTERRUPTED";

/************************************************ RUN ************************************************/

const int timer_L1_period = 1;      // [s] read water height period
const int timer_L2_period = 10;     // [s] date reading timer - PR2

Every timer_L1(timer_L1_period*1000);     // read water height timer
Every timer_L2(timer_L2_period*1000);     // date reading timer - PR2


/*********************************************** SD CARD ***********************************************/
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
  // Logger::print("Log set up.");

  // print_setup_summary(summary);
  // delay(5000);

  // // synchronize timers after setup
  timer_L2.reset(true);
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


/*********************************************** LOOP ***********************************************/ 
void loop() {

  if(timer_L1())
  {
    Logger::print("timer_L1 TICK");
    // Logger::printf(Logger::WARN, "timer_L1 TICK %f", 0.53f);
    // Serial.flush();
  }

  if(timer_L2())
  {
    // Logger::print("timer_L2 TICK");
    Logger::printf(Logger::WARN, "timer_L2 TICK %f\n", 0.53f);
    // Serial.flush();
  }
}