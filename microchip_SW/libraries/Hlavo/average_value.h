#ifndef AVERAGE_VALUE_H_
#define AVERAGE_VALUE_H_

#include "common.h"
#include <math.h>

class AverageValue {
public:
    explicit AverageValue(size_t window)
        : _cap(window), _buf(nullptr), _head(0), _count(0), _sum(0.0)
    {
        if (_cap == 0) _cap = 1;                // guard
        _buf = new float[_cap];
        // (optional) initialize memory, not required for logic
        for (size_t i = 0; i < _cap; ++i) _buf[i] = 0.0f;
    }

    ~AverageValue() { delete[] _buf; }

    // Add a new sample
    inline void add(float x) {
        if (_count < _cap) {
            _buf[_head] = x;
            _sum += x;
            _head = (_head + 1) % _cap;
            ++_count;
        } else {
            // overwrite oldest
            size_t idx = _head;
            float old = _buf[idx];
            _buf[idx] = x;
            _sum += (x - old);
            _head = (_head + 1) % _cap;
        }
    }

    // Return average; NaN until the window is full
    inline float average() const {
        if (_count < _cap)
          return std::numeric_limits<float>::quiet_NaN();
        return static_cast<float>(_sum / static_cast<double>(_cap));
    }

    // Return average; NaN until the window is full
    inline float average_unfilled() const {
        return static_cast<float>(_sum / static_cast<double>(_count));
    }

    // Clear all samples
    inline void reset() {
        _head = 0;
        _count = 0;
        _sum = 0.0;
        // (optional) zero the buffer if you like:
        // for (size_t i = 0; i < _cap; ++i) _buf[i] = 0.0f;
    }

    // Helpers
    inline bool isReady() const { return _count >= _cap; }
    inline size_t size()   const { return _count; }
    inline size_t capacity() const { return _cap; }

private:
    size_t  _cap;
    float*  _buf;
    size_t  _head;   // next write position
    size_t  _count;  // how many samples currently stored (<= _cap)
    double  _sum;    // use double to reduce accumulation error
};


#endif // AVERAGE_VALUE_H_
