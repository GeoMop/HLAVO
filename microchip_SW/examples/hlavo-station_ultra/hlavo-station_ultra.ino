#include <Every.h>
#include "clock.h"
// #include "Adafruit_I2CDevice.h"
#include "FS.h"
#include <SPI.h>


#define PIN_ON 47 // napajeni !!!

/** TIMERS */
// times in milliseconds, L*... timing level
Every timer_L1(1000); // fine timer - humidity, temperature, ...

Every timer_L2(5000); // fine timer - humidity, temperature, ...


/*********************************************** WATER HEIGHT ***********************************************/
#include "water_height_sensor.h"
WaterHeightSensor whs(5, 30, 220, 0.05, 3.13);  // pin, minH, maxH, minV, maxV

#define RELAY_PIN 15

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

  whs.begin();

  pinMode(RELAY_PIN, OUTPUT);      // Set EN pin for uSUP stabilisator as output
  digitalWrite(RELAY_PIN, HIGH);   // Turn on the uSUP power

  Serial.println("setup completed.");
  Serial.println(F("Start loop " __FILE__ " " __DATE__ " " __TIME__));
  Serial.println("--------------------------");

  // synchronize timers after setup
  timer_L2.reset(true);
  timer_L1.reset(true);
}


bool valve_open = false;

/*********************************************** LOOP ***********************************************/
void loop() {

  // read value to buffer at fine time scale
  if(timer_L1())
  {
    Serial.printf("L1 tick\n");
    float voltage;
    float height = whs.read(&voltage);
    Serial.printf("Voltage: %.2f    Height: %.2f\n", voltage, height);
  }


  if(timer_L2())
  {
    Serial.printf("L2 tick\n");
    if(valve_open)
    {
      Serial.printf("valve OFF\n");
      valve_open = false;
      digitalWrite(RELAY_PIN, HIGH);   // Turn on the uSUP power
    }
    else
    {
      Serial.printf("valve ON\n");
      valve_open = true;
      digitalWrite(RELAY_PIN, LOW);   // Turn on the uSUP power
    }
  }
}
