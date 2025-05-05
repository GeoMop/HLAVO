#include "WeatherMeters.h"

#define PIN_ON 47 // napajeni !!!



/****************************************** WHEATHER STATION ******************************************/
const int windvane_pin = A0; // A0 := 1
const int anemometer_pin = 5; 
const int raingauge_pin = 6; // 10 kOhm / 10pF

volatile bool got_data = false;

hw_timer_t * timer = NULL;
volatile SemaphoreHandle_t timerSemaphore;
volatile bool do_update = false;
portMUX_TYPE timerMux = portMUX_INITIALIZER_UNLOCKED;

WeatherMeters <4> meters(windvane_pin, 4);  // filter last 4 directions, refresh data every 8 sec

// ICACHE_RAM_ATTR replaced by IRAM_ATTR (esp and arduino>3.0.0)
void IRAM_ATTR intAnemometer() {
	meters.intAnemometer();
}

void IRAM_ATTR intRaingauge() {
	meters.intRaingauge();
}

void IRAM_ATTR onTimer() {
	xSemaphoreGiveFromISR(timerSemaphore, NULL);
	do_update = true;
}

void readDone(void) {
	got_data = true;
}


/*********************************************** SETUP ***********************************************/ 
void setup() {
  Serial.begin(115200);

  Serial.println("HLAVO project starts.");

  // for version over 3.5 need to turn uSUP ON
  Serial.print("set power pin: "); Serial.println(PIN_ON);
  pinMode(PIN_ON, OUTPUT);      // Set EN pin for uSUP stabilisator as output
  digitalWrite(PIN_ON, HIGH);   // Turn on the uSUP power

  pinMode(windvane_pin, ANALOG);
  pinMode(raingauge_pin, INPUT_PULLUP);  // Set GPIO as input with pull-up (like adding 10k resitor)
  pinMode(anemometer_pin, INPUT_PULLUP);  // Set GPIO as input with pull-up (like adding 10k resitor)
  attachInterrupt(digitalPinToInterrupt(anemometer_pin), intAnemometer, CHANGE);
	attachInterrupt(digitalPinToInterrupt(raingauge_pin), intRaingauge, CHANGE);

	meters.attach(readDone);

	timerSemaphore = xSemaphoreCreateBinary();
	timer = timerBegin(0, 80, true);
	timerAttachInterrupt(timer, &onTimer, true);
	timerAlarmWrite(timer, 1000000, true);
	timerAlarmEnable(timer);

	meters.reset();  // in case we got already some interrupts

  Serial.println("setup completed.");
}


/*********************************************** LOOP ***********************************************/ 
void loop() {
  if(do_update){
		meters.timer();
		do_update = false;
	}

	if (got_data) {
		got_data = false;

        //Serial.print("Směr větru: "); Serial.print(meters.getDir()); Serial.println(" deg");

        Serial.print("Wind direc adc: "); Serial.println(meters.getDirAdcValue());
        Serial.print("Wind direc deg: "); Serial.println(meters.getDir());
        Serial.print("Wind speed TICK: "); Serial.println(meters.getSpeedTicks());
        Serial.print("Rain gauge TICK: "); Serial.println(meters.getRainTicks());

        //Serial.print("Rychlost větru: "); Serial.print(meters.getSpeed()); Serial.println(" km/h"); 
        //Serial.print("Srážky: "); Serial.print(meters.getRain()); Serial.println(" mm");

        Serial.println("--------------------------");
   }
 
  delay(1);
}