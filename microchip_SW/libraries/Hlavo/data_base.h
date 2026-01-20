#ifndef HLAVO_DATA_BASE_H_
#define HLAVO_DATA_BASE_H_

#include <RTClib.h>
#include "common.h"

class DataBase
{
  public:
    DateTime datetime;
    static const char * delimiter;
    static const float NaN;
    // supposes maximal size of CSV line by hlavo::max_csvline_length
    virtual char* dataToCsvLine(char* csvLine) const;
    virtual char* dataToCsvLine(char* csvLine, size_t size) const = 0;

    DataBase()
    {
      datetime = DateTime(0,0,0, 0,0,0);
    }
};

const char * DataBase::delimiter = ";";
const float DataBase::NaN = 0.0;

char* DataBase::dataToCsvLine(char* csvLine) const
{
  return dataToCsvLine(csvLine, hlavo::max_csvline_length);
}

#endif // HLAVO_DATA_BASE_H_
