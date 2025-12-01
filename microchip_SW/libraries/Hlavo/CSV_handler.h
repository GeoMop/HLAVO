#pragma once

#include <RTClib.h>
#include "SD.h"
#include "file_info.h"
#include "data_base.h"

#include "common.h"
using namespace hlavo;

/// @brief Wrapper class for SD card and CSV file handling.
class CSVHandler
{
  public:
    static void createFile(char* user_filename, char* header, const DateTime& dt, const char* dir_name)
    {
      // date to string
      char dt_buf[20];
      sprintf(dt_buf, "YY-MM-DD_hh-mm-ss");
      dt.toString(dt_buf);

      char temp_filename[max_filepath_length];
      if(strlen(dir_name) >0){
        char dir_path[max_dirpath_length];
        snprintf(dir_path, sizeof(dir_path), "/%s", dir_name);
        SD.mkdir(dir_path);
        snprintf(temp_filename, sizeof(temp_filename), "/%s/%s_%s", dir_name, dt_buf, user_filename);
      }
      else
        snprintf(temp_filename, sizeof(temp_filename), "/%s_%s", dt_buf, user_filename);

      snprintf(user_filename, max_filepath_length, "%s", temp_filename);

      hlavo::SerialPrintf(200, "Creating file: %s\n", user_filename);
      FileInfo datafile(user_filename);
      datafile.write(header);
    }

    static void appendData(char* filename, DataBase* data)
    {
      char csvLine[max_csvline_length];
      FileInfo datafile(filename);
      data->dataToCsvLine(csvLine);
      datafile.append(csvLine);

      // File file = SD.open(filename, FILE_APPEND);
      // if(!file){
      //     Serial.println("Failed to open file for appending");
      // }
      // else
      // {
      //   data->dataToCsvLine(csvLine);
      //   // Serial.println(csvLine);

      //   bool res = file.print(csvLine);
      //   if(!res){
      //       Serial.println("Append failed");
      //   }
      // }
      // file.close();
    }

    static void appendData(char* filename, DataBase* data[], uint8_t n_data)
    {
      //Serial.printf("appendData: N %d\n", n_data);
      char csvLine[max_csvline_length];
      File file = SD.open(filename, FILE_APPEND);
      if(!file){
          Serial.println("Failed to open file for appending");
      }
      else
      {
        for(int i=0; i<n_data; i++)
        {
          // Serial.printf("appendData: %d\n", i);
          // Serial.printf("data: %s\n", data[i].datetime.timestamp().c_str());

          data[i]->dataToCsvLine(csvLine);
          // Serial.println(csvLine);

          bool res = file.print(csvLine);
          if(!res){
              Serial.println("Append failed");
          }
        }
      }
      file.close();
    }

    static void printFile(char* filename)
    {
      FileInfo datafile(filename);
      datafile.read();
    }
};