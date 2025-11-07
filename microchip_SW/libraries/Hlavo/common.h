#ifndef HLAVO_COMMON_H_
#define HLAVO_COMMON_H_

#include <stddef.h>
#include <stdlib.h>
#include <stdarg.h>


namespace hlavo{
  static const size_t max_dirpath_length = 100;
  static const size_t max_filepath_length = 200;
  static const size_t max_csvline_length = 400;

  static void strcat_safe(char* strdest, size_t size_dest, const char* str)
  {
    // test truncation
    // if (strlen(str) + 1 > size_dest - strlen(strdest))
            // Serial.print("onstack would be truncated");

    // keep the ending char '\0' at the end, throw away overflow
    (void)strncat(strdest, str, size_dest - strlen(strdest) - 1);
  }

  // Auxiliary Serial.sprintf function which is not available on Arduino
  static void SerialPrintf(uint16_t size, const char* format, ...)
  {
    char buf[size];

    va_list args;
    va_start(args, format);
    vsnprintf(buf, sizeof(buf), format, args);
    va_end(args);

    Serial.print(buf);
  }

  static void setPin(uint8_t pin, uint8_t state)
  {
    if(pin>0)
      digitalWrite(pin, state);
  }
};

#endif // HLAVO_COMMON_H_
