
#include <Arduino.h>
//#include "Every.h"

#define SERIAL_BAUD 115200 /*!< The baud rate for the output serial port */
#define DATA_PIN 8
#define POWER_PIN 22       /*!< The sensor power pin (or -1 if not switching power) */


void setup() {
  Serial.begin(SERIAL_BAUD);
  while (!Serial)
    ;

   pinMode(DATA_PIN, INPUT);

  // // Power the sensors;
  // if (POWER_PIN > 0) {
  //   Serial.println("Powering up sensors...");
  //   pinMode(POWER_PIN, OUTPUT);
  //   digitalWrite(POWER_PIN, HIGH);
  //   delay(500);//(200);
  // }
  Serial.println("Setup finished.");
}


int last = 0;
void loop() {
  
  
  int signal = digitalRead(DATA_PIN);

  if(signal != last)
  {
    Serial.println(signal);
    last = signal;
  }


  // if (mySerial.available()) { // Check if there is data available to read
  //   char c = mySerial.read();    // Read a single character
  //   Serial.println(c, HEX);
  // }
  // delay(20);

 
}