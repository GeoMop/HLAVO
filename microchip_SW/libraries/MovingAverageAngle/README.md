# Moving average filter for angles 0-360°
[![build](https://github.com/pilotak/MovingAverageAngle/workflows/build/badge.svg)](https://github.com/pilotak/MovingAverageAngle/actions) 
[![Framework Badge Arduino](https://img.shields.io/badge/framework-arduino-00979C.svg)](https://arduino.cc)
[![Framework Badge mbed](https://img.shields.io/badge/framework-mbed-008fbe.svg)](https://os.mbed.com/)

This library requires [MovingAverage](https://github.com/pilotak/MovingAverage) library.

## Arduino example
Please see `examples` folder

## Mbed example
```cpp
#include "mbed.h"
#include "MovingAverageAngle.h"  // https://github.com/pilotak/MovingAverageAngle

// Buffer will be 4 samples long, it will take 4 * sizeof(float) = 16 bytes of RAM
MovingAverageAngle <4> filter;

int main() {
    printf("result: %.2f\n", filter.add(350.0)); // insert new number and get result
    printf("result: %.2f\n", filter.add(20.0)); // insert new number and get result
    printf("result: %.2f\n", filter.add(30.0)); // insert new number and get result
    printf("result: %.2f\n", filter.add(40.0)); // insert new number and get result
    printf("result: %.2f\n", filter.get()); // get last result, without adding a newone

    return 0;
}
```

### Output
> result: 350.00
> 
> result: 357.37
> 
> result: 7.47
> 
> result: 20.24
> 
> result: 20.24
