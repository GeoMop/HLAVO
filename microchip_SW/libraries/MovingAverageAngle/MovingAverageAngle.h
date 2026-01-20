/*
MIT License

Copyright (c) 2018 Pavel Slama

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef MOVINGAVERAGEANGLE_H
#define MOVINGAVERAGEANGLE_H

#if defined(__MBED__)
    #include "mbed.h"
#endif

#include <math.h>
#include <MovingAverage.h>

#if !defined(DEG_TO_RAD)
    #define DEG_TO_RAD 0.017453292519943295769236907684886
#endif

#if !defined(RAD_TO_DEG)
    #define RAD_TO_DEG 57.295779513082320876798154814105
#endif

template <uint8_t N>
class MovingAverageAngle {
  public:
    MovingAverageAngle();
    virtual ~MovingAverageAngle(void);

    double add(double value);
    double get();
    void fill(double value);
    void reset();

  private:
    MovingAverage <int32_t, N> _filterSin;
    MovingAverage <int32_t, N> _filterCos;

    double toRadian(double value);

    double _result;
};

template <uint8_t N>
MovingAverageAngle<N>::MovingAverageAngle() {
    _result = 0;
}

template <uint8_t N>
MovingAverageAngle<N>::~MovingAverageAngle(void) {
}

template <uint8_t N>
double MovingAverageAngle<N>::get() {
    return _result;
}

template <uint8_t N>
double MovingAverageAngle<N>::add(double value) {
    double radian = toRadian(value);
    double s = static_cast<double>(_filterSin.add(static_cast<int32_t>(sin(radian) * 10000000))) / 10000000.0;
    double c = static_cast<double>(_filterCos.add(static_cast<int32_t>(cos(radian) * 10000000))) / 10000000.0;
    double deg = atan2(s, c) * RAD_TO_DEG;

    if (deg < 0) deg += 360.0;

    if (deg >= 360.0) deg = 0.0;

    _result = deg;

    return _result;
}

template <uint8_t N>
void MovingAverageAngle<N>::fill(double value) {
    double radian = toRadian(value);

    _filterSin.fill(static_cast<int32_t>(sin(radian) * 10000000));
    _filterCos.fill(static_cast<int32_t>(cos(radian) * 10000000));
}

template <uint8_t N>
void MovingAverageAngle<N>::reset() {
    _filterSin.reset();
    _filterCos.reset();
}

template <uint8_t N>
double MovingAverageAngle<N>::toRadian(double value) {
    return value * DEG_TO_RAD;
}

#endif
