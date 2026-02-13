#ifndef RAIN_REGIME_H_
#define RAIN_REGIME_H_

#include "common.h"
#include "clock.h"
#include "Every.h"
#include <math.h>

class RainRegime{
  public:
    static const float pump_rate;       // [l/min]
    static const float column_radius;   // [m]
    static const float column_cross;    // [m2]

    static Every timer_period;
    static Timer timer_length;

    unsigned long trigger_length; // [s]
    unsigned long trigger_period; // [h]
    unsigned long length;         // [h]
    unsigned long wait;           // [h]

    DateTime last_rain;
    uint8_t counter;

    // [s], [h], [h], [h]
    RainRegime(unsigned long tr_legnth_s, float tr_period_h, float length_h, float wait_h)
    {
      trigger_length = tr_legnth_s * 1000;
      trigger_period = (unsigned long) (tr_period_h * 3600 * 1000);

      length = (unsigned long) (length_h * 3600 * 1000);
      wait = (unsigned long) (wait_h * 3600 * 1000);

      last_rain = DateTime((uint32_t)0);
      counter = 0;
    }

    void reset_timer_length(DateTime dtnow)
    {
      timer_length.reset(trigger_length);
      last_rain = dtnow;
      counter++;
    }

    void reset_timer_period()
    {
      timer_period.reset(trigger_period);
    }

    char* print(char* msg_buf, size_t size) const
    {
      snprintf(msg_buf,  size,
              "Rain Regime: "
              "tl %d, "
              "tp %d, "
              "L %d, "
              "W %d\n",
              trigger_length/1000,
              trigger_period/1000,
              length/1000,
              wait/1000);
      return msg_buf;
    }
};

const float RainRegime::pump_rate = 1/(3.0+43.0/60.0);            // [l/min], measurement 1l in 3:43 min
const float RainRegime::column_radius = 0.15;        // [m]
const float RainRegime::column_cross = PI * column_radius*column_radius;  // [m2]

Every RainRegime::timer_period = Every(1000);
Timer RainRegime::timer_length = Timer(1000);

#endif // RAIN_REGIME_H_
