#include <Every.h>
#include "clock.h"  // do not know why SPI and Adafruit is missing in a build


#define PIN_ON 47 // napajeni !!!

/** TIMERS */
// times in milliseconds, L*... timing level
Every timer_L1(1000); // fine timer


/*********************************************** WATER HEIGHT ***********************************************/
#include "linear_voltage_sensor.h"
//    0 mm: 630 mm - 2.21 V : water height of filled column (10 mm below top)
// -240 mm: 390 mm - 1.33 V
// -620 mm:  10 mm - 0.05 V
LinearVoltageSensor whs(7, 0.05, 2.21, 10, 630);  // pin, aVolt, bVolt, aVal, bVal
float minimal_water_height = 100; // mm

// diameter
// column: 120 mm
// hadice: 12 mm
// cable: 6 mm
// hadicka: 6 mm (ale jen tloustka)

/************************************************ RAIN *************************************************/
#define PUMP_IN_PIN 6
bool pump_in_finished = true;
Timer timer_rain_start(10*1000, false);    // timer before rain


void start_rain()
{
  // start rain
  digitalWrite(PUMP_IN_PIN, LOW);
  Serial.printf("rain ON\n");
  pump_in_finished = false;
}

void stop_rain()
{
    // stop rain
  digitalWrite(PUMP_IN_PIN, HIGH);
  Serial.printf("rain OFF\n");
  pump_in_finished = true;
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

  whs.setWindow(0.0,3.0);
  whs.begin();

  Serial.println("setup completed.");
  Serial.println(F("Start loop " __FILE__ " " __DATE__ " " __TIME__));
  Serial.println("--------------------------");

  // synchronize timers after setup
  timer_L1.reset(true);
  timer_rain_start.reset();
}


unsigned int time_counter = 0;
unsigned int counter = 0;
float voltage_sum = 0.0;

/*********************************************** LOOP ***********************************************/
void loop() {

  if(timer_rain_start())
  {
    start_rain();
    time_counter = 0;
  }

  // read value to buffer at fine time scale
  if(timer_L1())
  {
    Serial.printf("L1 tick\n");
    float voltage;
    float height = whs.read(&voltage);

    voltage_sum += voltage;
    counter ++;

    Serial.printf("Voltage: %.2f V    Height: %.2f mm\n", voltage, height);
    // Serial.printf("Voltage: %.2f V   Height: %.2f mm   Volt_Avg: %.2f V (n=%d)\n", voltage, height, voltage_sum/counter, counter);

    if(!pump_in_finished)
      time_counter++;

    if(height < minimal_water_height) // mm
    {
      stop_rain();
      Serial.printf("Rain length: %d s\n", time_counter);
    }
  }
}
