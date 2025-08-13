# pragma once

#include <RTClib.h> // just for DateTime
#include "data_base.h"
#include "common.h"
using namespace hlavo;

class ColumnFlowData : public DataBase{

  public:
    float height;
    float flux;
    uint8_t pump_in;
    uint8_t pump_out;
    float rain_height;

    static char* headerToCsvLine(char* csvLine, size_t size);
    ColumnFlowData();
    // Function to convert ColumnFlowData struct to CSV string with a custom delimiter
    char* dataToCsvLine(char* csvLine, size_t size) const override;
    // Print ColumnFlowData
    char* print(char* msg_buf, size_t size) const;
};


char* ColumnFlowData::headerToCsvLine(char* csvLine, size_t size){
    const int n_columns = 6;
    const char* columnNames[] = {
      "DateTime",
      "Height",
      "Flux",
      "PumpIn",
      "PumpOut",
      "RainHeight"
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

ColumnFlowData::ColumnFlowData()
  : DataBase()
{
  height = std::numeric_limits<float>::quiet_NaN();
  flux = std::numeric_limits<float>::quiet_NaN();
  pump_in = 0;
  pump_out = 0;
  rain_height = std::numeric_limits<float>::quiet_NaN();
}

char* ColumnFlowData::dataToCsvLine(char* csvLine, size_t size) const
{
  // Serial.println("ColumnFlowData::dataToCsvLine");
  const char * dt = datetime.timestamp().c_str();
  snprintf(csvLine, size,
          "%s%s"    // datetime
          "%.2f%s"  // height
          "%.2f%s"  // flux
          "%d%s"    // pump in
          "%d%s"    // valve out
          "%.2f\n", // rain height
          dt, delimiter,
          height, delimiter,
          flux, delimiter,
          pump_in, delimiter,
          pump_out, delimiter,
          rain_height);
  return csvLine;
}

char* ColumnFlowData::print(char* msg_buf, size_t size) const
{
  snprintf(msg_buf,  size,
          "%s   "
          "Height %.2f, " // height
          "Flux %.2f, "   // flux
          "PumpIn %d, "   // pump in
          "PumpOut %d, "  // valve out
          "RainHeight %.2f", // rain height
          datetime.timestamp().c_str(),
          height,
          flux,
          pump_in,
          pump_out,
          rain_height);
  return msg_buf;
}
