#include <Every.h>
#include "clock.h"  // do not know why SPI and Adafruit is missing in a build


#define PIN_ON 47 // napajeni !!!

/** TIMERS */
// times in milliseconds, L*... timing level
Every timer_L1(1000); // fine timer


/*********************************************** WATER HEIGHT ***********************************************/
#include "linear_voltage_sensor.h"

// ultrasonic sensor S18U
// LinearVoltageSensor whs(5, 0.05, 3.13, 220, 30);  // pin, aVolt, bVolt, aVal, bVal

//    0 mm: 630 mm - 2.21 V : water height of filled column (10 mm below top)
// -240 mm: 390 mm - 1.33 V
// -620 mm:  10 mm - 0.05 V
LinearVoltageSensor whs(7, 0.05, 2.21, 10, 630);  // pin, aVolt, bVolt, aVal, bVal
// radius:
// column: 60 mm
// hadice: 6 mm
// cable: 3 mm
// hadicka: 3 mm (ale jen tloustka)


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

  // setup water height sensor
  whs.setWindow(0.0,3.0);
  whs.begin();

  Serial.println("setup completed.");
  Serial.println(F("Start loop " __FILE__ " " __DATE__ " " __TIME__));
  Serial.println("--------------------------");

  // synchronize timers after setup
  timer_L1.reset(true);
}


unsigned int counter = 0;
float voltage_sum = 0.0;

/*********************************************** LOOP ***********************************************/
void loop() {

  // read value to buffer at fine time scale
  if(timer_L1())
  {
    Serial.printf("L1 tick\n");
    float voltage;
    float height = whs.read(&voltage);

    voltage_sum += voltage;
    counter ++;

    // Serial.printf("Voltage: %.2f V    Height: %.2f mm\n", voltage, height);
    Serial.printf("Voltage: %.2f V   Height: %.2f mm   Volt_Avg: %.2f (n=%d)\n", voltage, height, voltage_sum/counter, counter);
  }
}
