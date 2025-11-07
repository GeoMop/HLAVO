
/*********************************************** COMMON ***********************************************/
#include <Every.h>

#define PIN_ON 47 // power stabilizer

const char* setup_interrupt = "SETUP INTERRUPTED";

/************************************************ RUN ************************************************/
/** TIMERS */
// times in milliseconds, L*... timing level
Every timer_L1(1000);
Every timer_L2(5000);
Every timer_L4(10*60*1000);  // watchdog timer - 10 min


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
      ; // wait for Serial port
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

  // while(1){delay(100);}
  delay(1000);

  // synchronize timers after setup
  timer_L2.reset(true);
  timer_L1.reset(true);
  timer_L4.reset(false);

  print_setup_summary(summary);
  delay(2000);
}

void print_setup_summary(String summary)
{
  summary = "\nSETUP SUMMARY:\n" + summary;
  summary = "\n=======================================================================\n" + summary + "\n";
  summary += F("INO file: " __FILE__ " " __DATE__ " " __TIME__ "\n\n");
  summary += "=======================================================================";

  Serial.print(summary); Serial.println("");
  Serial.flush();
}

/*********************************************** LOOP ***********************************************/ 
void loop() {
  
  if(timer_L1())
  {
    Serial.printf("        -------------------------- L1 TICK -------------------------- till L2: %d s\n",
      (timer_L2.interval + timer_L2.last - millis())/1000);
    
    DateTime dt = rtc_clock.now();
    Serial.printf("%s\n", dt.timestamp().c_str());
  }

  if(timer_L2())
  {
    Serial.println("    **************************************** L2 TICK ****************************************");   
  }

  if(timer_L4())
  {
    Serial.println("-------------------------- L4 TICK --------------------------");
    Serial.printf("\nReboot...\n\n");
    delay(250);
    ESP.restart();
  }
}