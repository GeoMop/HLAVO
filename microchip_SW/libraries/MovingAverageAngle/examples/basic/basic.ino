#include "MovingAverageAngle.h"  // https://github.com/pilotak/MovingAverageAngle

// Buffer will be 4 samples long, it will take 4 * sizeof(float) = 16 bytes of RAM
MovingAverageAngle <4> filter;

void setup() {
    Serial.begin(9600);
    Serial.print("result: ");
    Serial.println(filter.add(350.0), 2); // insert new number and get result
    Serial.print("result: ");
    Serial.println(filter.add(20.0), 2); // insert new number and get result
    Serial.print("result: ");
    Serial.println(filter.add(30.0), 2); // insert new number and get result
    Serial.print("result: ");
    Serial.println(filter.add(40.0), 2); // insert new number and get result
    Serial.print("result: ");
    Serial.println(filter.get(), 2); // get last result, without adding a newone
}

void loop() {

}