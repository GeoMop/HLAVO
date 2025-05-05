# pragma once

#include <RTClib.h> // just for DateTime
#include "data_base.h"
#include "common.h"
using namespace hlavo;

class BME280Data : public DataBase{

  public:
    float humidity;
    float pressure;
    float temperature;

    static char* headerToCsvLine(char* csvLine, size_t size);
    BME280Data();
    // Function to convert BME280Data struct to CSV string with a custom delimiter
    char* dataToCsvLine(char* csvLine, size_t size) const override;
    // Print BME280Data
    char* print(char* msg_buf, size_t size) const;
};


char* BME280Data::headerToCsvLine(char* csvLine, size_t size){
    const int n_columns = 4;
    const char* columnNames[] = {
      "DateTime",
      "Humidity",
      "Pressure",
      "Temperature"
    };
    csvLine[0] = '\0'; // Initialize the CSV line as an empty string

    // Iterate through the array of strings
    for (int i = 0; i < n_columns; ++i) {
        // Concatenate the current string to the CSV line
        strcat_safe(csvLine, size, columnNames[i]);

        // If it's not the last string, add the delimiter
        if (i < n_columns - 1)
          strcat_safe(csvLine, size, delimiter);
        else
          strcat_safe(csvLine, size, "\n");
    }

    return csvLine;
}

BME280Data::BME280Data()
  : DataBase()
{
  humidity = std::numeric_limits<float>::quiet_NaN();
  pressure = std::numeric_limits<float>::quiet_NaN();
  temperature = std::numeric_limits<float>::quiet_NaN();
}

char* BME280Data::dataToCsvLine(char* csvLine, size_t size) const
{
  // Serial.println("BME280Data::dataToCsvLine");
  const char * dt = datetime.timestamp().c_str();
  snprintf(csvLine, size,
          "%s%s"    // datetime
          "%.2f%s"  // humidity
          "%.2f%s"  // pressure
          "%.2f\n", // temperature
          dt, delimiter,
          humidity, delimiter,
          pressure, delimiter,
          temperature);
  return csvLine;
}

char* BME280Data::print(char* msg_buf, size_t size) const
{
  snprintf(msg_buf,  size,
          "%s   "
          "Humidity %.2f, "   // humidity
          "Pressure %.2f, "   // pressure
          "Temperature %.2f",   // temperature
          datetime.timestamp().c_str(),
          humidity,
          pressure,
          temperature);
  return msg_buf;
}
