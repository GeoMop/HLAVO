# pragma once

#include <RTClib.h> // just for DateTime
#include "data_base.h"
#include "common.h"
using namespace hlavo;

#define NUM_FINE_VALUES 4         // humidity, temperature, light_mean, battery_voltage
#define FINE_DATA_BUFSIZE 70      // 60 per 1 min
#define METEO_DATA_BUFSIZE 20     // 15 per 15 min

class MeteoData : public DataBase{

  public:
    // weather station
    float wind_direction;
    float wind_speed;
    float raingauge;

    // RTC
    float humidity_mean, humidity_var;
    float temperature_mean, temperature_var;

    float light_mean, light_var;

    // battery - ESP32 analog read
    float battery_voltage_mean, battery_voltage_var;


    static char* headerToCsvLine(char* csvLine, size_t size);
    MeteoData();
    // Function to convert MeteoData struct to CSV string with a custom delimiter
    char* dataToCsvLine(char* csvLine, size_t size) const override;
    void compute_statistics(float fineValues[][FINE_DATA_BUFSIZE], u16_t size_m, u16_t size_n);
    // Print MeteoData
    char* print(char* msg_buf, size_t size) const;
};


char* MeteoData::headerToCsvLine(char* csvLine, size_t size){
    const int n_columns = 12;
    const char* columnNames[] = {
      "DateTime",
      "WindDirection",
      "WindSpeed",
      "RainGauge",
      "Humidity_Mean",
      "Humidity_Var",
      "Temperature_Mean",
      "Temperature_Var",
      "Light_Mean",
      "Light_Var",
      "BatteryVoltage_Mean",
      "BatteryVoltage_Var"
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

MeteoData::MeteoData()
  : DataBase()
{
  wind_direction = 0.0f;
  wind_speed = 0.0f;
  raingauge = 0.0f;

  humidity_mean = 0.0f;
  humidity_var = 0.0f;
  temperature_mean = 0.0f;
  temperature_var = 0.0f;
  light_mean = 0.0f;
  light_var = 0.0f;

  battery_voltage_mean = 0.0f;
  battery_voltage_var = 0.0f;
}

char* MeteoData::dataToCsvLine(char* csvLine, size_t size) const
{
  const char * dt = datetime.timestamp().c_str();
  // Format datetime in "YYYY-MM-DD HH:MM:SS" format
  // sprintf(datetime, "%04d-%02d-%02d %02d:%02d:%02d%c%.2f%c%u%c%u%c%.2f%c%.2f%c%.2f\n",
  //         data.datetime.year(), data.datetime.month(), data.datetime.day(),
  //         data.datetime.hour(), data.datetime.minute(), data.datetime.second());
  snprintf(csvLine, size,
          "%s%s"    // datetime
          "%.1f%s"  // wind direction
          "%.2f%s"  // wind speed
          "%.2f%s"  // raingauge
          "%.2f%s%.2f%s" // humidity
          "%.2f%s%.2f%s" // temperature
          "%.0f%s%.0f%s" // light
          "%.3f%s%.3f\n",// battery
          dt, delimiter,
          wind_direction, delimiter,
          wind_speed, delimiter,
          raingauge, delimiter,
          humidity_mean, delimiter,
          humidity_var, delimiter,
          temperature_mean, delimiter,
          temperature_var, delimiter,
          light_mean, delimiter,
          light_var, delimiter,
          battery_voltage_mean, delimiter,
          battery_voltage_var);
  return csvLine;
}

void MeteoData::compute_statistics(float fineValues[][FINE_DATA_BUFSIZE], u16_t size_m, u16_t size_n)
{
  float diff;
  float mean[size_m];
  float var[size_m];

  for (int i = 0; i < size_m; i++) {
    mean[i] = 0;
    // Calculate mean
    for (int j = 0; j < size_n; j++) {
      mean[i] += fineValues[i][j];
    }
    mean[i] /= size_n;

    var[i] = 0;
    // Calculate variance
    for (int j = 0; j < size_n; j++) {
      diff = fineValues[i][j] - mean[i];
      var[i] += diff*diff;
    }
    var[i] /= size_n;
  }

  humidity_mean = mean[0];
  temperature_mean = mean[1];
  light_mean = mean[2];
  battery_voltage_mean = mean[3];

  humidity_var = var[0];
  temperature_var = var[1];
  light_var = var[2];
  battery_voltage_var = var[3];
}

char* MeteoData::print(char* msg_buf, size_t size) const
{
  snprintf(msg_buf,  size,
          "%s   "
          "WindDir %.1f, "  // wind direction
          "WindSpd %.2f, "  // wind speed
          "Rain %.2f,  "  // raingauge
          "Hum %.2f, " // humidity
          "Temp %.2f, " // temperature
          "Light %.0f, " // light
          "Bat %.3f",// battery
          datetime.timestamp().c_str(),
          wind_direction,
          wind_speed,
          raingauge,
          humidity_mean,
          temperature_mean,
          light_mean,
          battery_voltage_mean);
  return msg_buf;
}
