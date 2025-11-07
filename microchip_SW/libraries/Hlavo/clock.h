#ifndef HLAVO_CLOCK_H_
#define HLAVO_CLOCK_H_

#if defined(ESP32) || defined(ESP8266)
  #include <ESP32Time.h>
  #include "Adafruit_Sensor.h"
#endif

#include <RTClib.h>
#include "common.h"

class Clock {
  private:
    RTC_DS3231 rtc;
#if defined(ESP32) || defined(ESP8266)
    ESP32Time internal_rtc;
#endif


  public:
    // Constructor
    Clock()
    {}

    // Initialize the clock
    bool begin() {
      Serial.println(F("Initializing RTC..."));

      if (!rtc.begin()) {
        return false;
      }
      if (rtc.lostPower())
      {
        DateTime dt = DateTime(F(__DATE__), F(__TIME__));
        // UTC time: (Prague - 1) (summer time)
        dt = dt - TimeSpan(3600);
        hlavo::SerialPrintf(100,"RTC lost power. Setting time: %s\n", dt.timestamp().c_str());
        // When time needs to be set on a new device, or after a power loss, the
        // following line sets the RTC to the date & time this sketch was compiled
        rtc.adjust(dt);//rtc.adjust(DateTime(2024, 3, 01, 10, 15, 0));
        // This line sets the RTC with an explicit date & time, for example to set
        // January 21, 2014 at 3am you would call:
        // rtc.adjust(DateTime(2014, 1, 21, 3, 0, 0));
      }
      // rtc.adjust(DateTime(2024, 6, 17, 9, 53, 0));

#if defined(ESP32) || defined(ESP8266)
      // Update internal RTC in ESP chip due to correct filesystem datetime
      internal_rtc.setTime(rtc.now().unixtime());
#endif

      return true;
    }

    RTC_DS3231& get_rtc()
    {
      return rtc;
    }

    // Get the current date and time from the clock
    DateTime now() {
      return rtc.now();
    }
};

#endif // HLAVO_CLOCK_H_
