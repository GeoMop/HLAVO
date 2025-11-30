
/*********************************************** COMMON ***********************************************/
#include <Every.h>
#include <Logger.h>
#include <math.h>

#define PIN_ON 47 // napajeni !!!

const char* setup_interrupt = "SETUP INTERRUPTED";

/************************************************ RUN ************************************************/
// Switch between testing/setup and long term run.

/** TIMERS */
// times in milliseconds, L*... timing level
const int timer_L0_period = 1;        // [s] time reader
const int timer_L2_period = 15;     // [s] date reading timer - PR2

Every timer_L0(timer_L0_period*1000);     // time reader
Every timer_L2(timer_L2_period*1000);     // date reading timer - PR2


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


/********************************************** SDI12 COMM ********************************************/
#include "sdi12_comm.h"
#define SDI12_DATA_PIN 4         // The pin of the SDI-12 data bus

SDI12Comm sdi12_comm(SDI12_DATA_PIN, 1);  // (data_pin, verbose)

/********************************************* PR2 SENSORS ********************************************/
#include "pr2_data.h"
#include "pr2_reader.h"

const char pr2_address = '3';  // sensor addresses on SDI-12
PR2Reader pr2_reader = PR2Reader(&sdi12_comm, pr2_address);
char data_pr2_filename[max_filepath_length] = {"pr2_a3_calib.csv"};

bool pr2_all_finished = false;


/****************************************** DATA COLLECTION ******************************************/

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
  // while(1){delay(1000);}

  // Data files setup
  char csvLine[400];
  
  // PR2 data file
  const char* pr2_dir="pr2_sensor";
  CSVHandler::createFile(data_pr2_filename,
                         PR2Data::headerToCsvLine(csvLine, max_csvline_length),
                         dt_now, pr2_dir);

  delay(500);
  print_setup_summary(summary);
  // delay(5000);

  // synchronize timers after setup
  timer_L2.reset(true);

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


/*********************************************** LOOP ***********************************************/ 
void loop() {

  if(timer_L0()){
    dt_now = rtc_clock.now();
  }

  // do not read data during rain (voltage source damages data signal)
  {
    // read values from PR2 and Teros31 sensors when reading not finished yet
    // and write to a file when last values received
    if(!pr2_all_finished)
      collect_and_write_PR2();

    if(timer_L2())
    {
      Serial.println("-------------------------- L2 TICK --------------------------");

      if(pr2_all_finished){
        pr2_all_finished = false;
      }
    }
  }
}