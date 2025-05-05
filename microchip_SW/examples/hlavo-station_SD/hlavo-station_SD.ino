// SD card IO
#include "Adafruit_SHT4x.h"
#include "SD.h"
// file handling
#include "file_info.h"


#define PIN_ON 47 // napajeni !!!
// SD card pin
#define SD_CS_PIN 10

#define data_meteo_filename "/meteo.txt"

 
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

    while (!Serial)
    {
        ; // cekani na Serial port
    }

    FileInfo datafile(data_meteo_filename);
    datafile.append("World!\n");
    datafile.read();
}
 
void loop() {

  FileInfo datafile(data_meteo_filename);

  char buffer[100];
  ltoa(millis(), buffer, 10);
  datafile.append(buffer);
  datafile.append("\n");
  Serial.println(buffer);
  Serial.println("--------------------------");

  delay(2000);
}