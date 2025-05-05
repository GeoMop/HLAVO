#ifndef TEROS31_DATA_H_
#define TEROS31_DATA_H_

#include "common.h"
#include "data_base.h"

using namespace hlavo;

/// @brief Data class for handling PR2 data from a single sensor.
class Teros31Data : public DataBase{
  public:
    float temperature;
    float pressure;

    static char* headerToCsvLine(char* csvLine, size_t size);

    Teros31Data();
    char* dataToCsvLine(char* csvLine, size_t size) const override;
    char* print(char* msg_buf, size_t size) const;
};


char* Teros31Data::headerToCsvLine(char* csvLine, size_t size) {

  csvLine[0] = '\0'; // Initialize the CSV line as an empty string
  
  strcat_safe(csvLine, size, "DateTime");
  strcat_safe(csvLine, size, delimiter);
  strcat_safe(csvLine, size, "Temperature");
  strcat_safe(csvLine, size, delimiter);
  strcat_safe(csvLine, size, "Pressure");
  strcat_safe(csvLine, size, "\n");

  return csvLine;
}

Teros31Data::Teros31Data()
  : DataBase()
{
  temperature = 0;
  pressure = 0;
}

// Function to convert PR2Data struct to CSV string with a custom delimiter
char* Teros31Data::dataToCsvLine(char* csvLine, size_t size) const {

  const char * dt = datetime.timestamp().c_str();
  snprintf(csvLine, size, "%s%s", dt, delimiter);
  char number[10];

  snprintf(number, sizeof(number), "%.4f%s", temperature, delimiter);
  strcat_safe(csvLine, size, number);
  snprintf(number, sizeof(number), "%.4f", pressure);
  strcat_safe(csvLine, size, number);
  strcat_safe(csvLine, size, "\n");

  return csvLine;
}

// Print PR2Data
char* Teros31Data::print(char* msg_buf, size_t size) const {

  const char * dt = datetime.timestamp().c_str();
  snprintf(msg_buf, size, "%s ", dt);
  char number[10];

  strcat_safe(msg_buf, size, "Press.: ");
  snprintf(number, sizeof(number), "%.4f", pressure);
  strcat_safe(msg_buf, size, number);

  strcat_safe(msg_buf, size, " Temp.: ");
  snprintf(number, sizeof(number), "%.4f", temperature);
  strcat_safe(msg_buf, size, number);

  return msg_buf;
}

#endif // TEROS31_DATA_H_
