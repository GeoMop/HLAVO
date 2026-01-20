// #define ENABLE_ARDUINO_IDE_SUPPORT
 
// #ifdef ENABLE_ARDUINO_IDE_SUPPORT
// #include "customLibs.h"
// #else
// sbernice
#include <Wire.h>
// temperature, humidity
#include "Adafruit_SHT4x.h"
// communication (sbernice)
#include <SPI.h>
// read voltage of ESP32 battery
#include "ESP32AnalogRead.h"
#include "WeatherMeters.h"
// real time
#include "RTClib.h"
// SD card IO
#include "SD.h"
// file handling
#include "file_info.h"
// #endif

// defnice sbernice i2C a SPI
#define SDA 42
#define SCL 2
#define PIN_ON 47 // napajeni !!!
// SD card pin
#define SD_CS_PIN 10

#define data_meteo_filename "/meteo.txt"

RTC_DS3231 rtc;
char daysOfTheWeek[7][12] = {"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"};

ESP32AnalogRead adc;
#define ADCpin 9
#define DeviderRatio 1.7693877551  // Voltage devider ratio on ADC pin 1MOhm + 1.3MOhm


Adafruit_SHT4x sht4 = Adafruit_SHT4x();

const int windvane_pin = A0; 
const int anemometer_pin = 5; 
const int raingauge_pin = 6; // 10 kOhm / 10pF

volatile bool got_data = false;

hw_timer_t * timer = NULL;
volatile SemaphoreHandle_t timerSemaphore;
volatile bool do_update = false;
portMUX_TYPE timerMux = portMUX_INITIALIZER_UNLOCKED;

WeatherMeters <4> meters(windvane_pin, 8);  // filter last 4 directions, refresh data every 8 sec

void ICACHE_RAM_ATTR intAnemometer() {
	meters.intAnemometer();
}

void ICACHE_RAM_ATTR intRaingauge() {
	meters.intRaingauge();
}

void IRAM_ATTR onTimer() {
	xSemaphoreGiveFromISR(timerSemaphore, NULL);
	do_update = true;
}

void readDone(void) {
	got_data = true;
}

 
void setup() {
    Serial.begin(115200);

    Serial.println("HLAVO project starts.");

    // for version over 3.5 need to turn uSUP ON
    pinMode(PIN_ON, OUTPUT);      // Set EN pin for uSUP stabilisator as output
    digitalWrite(PIN_ON, HIGH);   // Turn on the uSUP power

    Serial.println("HLAVO project starts.");


    pinMode(SD_CS_PIN, OUTPUT);
    // SD Card Initialization
    if (SD.begin()){
        Serial.println("SD card is ready to use.");
    }
    else{
        Serial.println("SD card initialization failed");
        return;
    }

    adc.attach(ADCpin); // setting ADC

    attachInterrupt(digitalPinToInterrupt(anemometer_pin), intAnemometer, FALLING);
	attachInterrupt(digitalPinToInterrupt(raingauge_pin), intRaingauge, FALLING);

	meters.attach(readDone);

	timerSemaphore = xSemaphoreCreateBinary();
	timer = timerBegin(0, 80, true);
	timerAttachInterrupt(timer, &onTimer, true);
	timerAlarmWrite(timer, 1000000, true);
	timerAlarmEnable(timer);

	meters.reset();  // in case we got already some interrupts

    while (!Serial)
    {
        ; // cekani na Serial port
    }

    Wire.begin(SDA, SCL);

    if (! sht4.begin())
    {
        Serial.println("SHT4x nenalezen");
        Serial.println("Zkontrolujte propojeni");
        while (1) delay(1);
    }

    sht4.setPrecision(SHT4X_HIGH_PRECISION); // nejvyssi rozliseni
    sht4.setHeater(SHT4X_NO_HEATER); // bez vnitrniho ohrevu

    if (! rtc.begin()) {
        Serial.println("RTC nenalezen");
        Serial.flush();
        while (1) delay(10);
    }

    if (rtc.lostPower()) {
        Serial.println("RTC lost power, let's set the time!");
        // When time needs to be set on a new device, or after a power loss, the
        // following line sets the RTC to the date & time this sketch was compiled
        rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));//rtc.adjust(DateTime(2024, 3, 01, 10, 15, 0));
        // This line sets the RTC with an explicit date & time, for example to set
        // January 21, 2014 at 3am you would call:
        // rtc.adjust(DateTime(2014, 1, 21, 3, 0, 0));
    }

    // When time needs to be re-set on a previously configured device, the
    // following line sets the RTC to the date & time this sketch was compiled
    //rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));
    // This line sets the RTC with an explicit date & time, for example to set
    // January 21, 2014 at 3am you would call:
    // rtc.adjust(DateTime(2014, 1, 21, 3, 0, 0));
    FileInfo datafile(SD, data_meteo_filename);
    datafile.append("World!\n");
    datafile.read();
}
 
void loop() {
    sensors_event_t humidity, temp; // promenne vlhkost a teplota

    if(do_update){
        meters.timer();
        do_update = false;
    }

    if (got_data) {
        got_data = false;

        DateTime now = rtc.now();

        Serial.print(now.day(), DEC); Serial.print('/');
        Serial.print(now.month(), DEC); Serial.print('/');
        Serial.print(now.year(), DEC);
        //Serial.print(" ("); Serial.print(daysOfTheWeek[now.dayOfTheWeek()]); Serial.print(")");
        Serial.print(" ");
        Serial.print(now.hour(), DEC);
        Serial.print(':');
        Serial.print(now.minute(), DEC);
        Serial.print(':');
        Serial.print(now.second(), DEC);
        Serial.println();

        Serial.print("Směr větru: "); Serial.print(meters.getDir()); Serial.println(" deg");

        Serial.print("Rychlost větru TICK: "); Serial.println(meters.getSpeedTicks());
        Serial.print("Srážky TICK: "); Serial.println(meters.getRainTicks());

        Serial.print("Rychlost větru: "); Serial.print(meters.getSpeed()); Serial.println(" km/h");
        Serial.print("Srážky: "); Serial.print(meters.getRain()); Serial.println(" mm");

        sht4.getEvent(&humidity, &temp);
        Serial.print("Teplota: "); Serial.print(temp.temperature); Serial.println(" degC");
        Serial.print("Vlhkost: "); Serial.print(humidity.relative_humidity); Serial.println("% rH");

        float bat_voltage = adc.readVoltage() * DeviderRatio;
        Serial.print("Baterie: "); Serial.print(bat_voltage); Serial.println(" V");
        Serial.println("--------------------------");
    }
 
    delay(500);

    // TODO:
    // zapis meteo hodnot
    // zapis 2 sond PR2 (JB: jake hodnoty)
    // platformio
    // konverze tabulky pro smer vetru (5V->3.3V)
    // kalibrace rychlost vetru
    // vyzkouset, jak se chova vyndavani karty, co je potreba k pokracovani
}