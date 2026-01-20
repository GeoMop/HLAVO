#ifndef TEROS31_READER_H_
#define TEROS31_READER_H_

#include "sdi12_comm.h"
#include "common.h"
#include "teros31_data.h"

using namespace hlavo;

/// @brief Class that keeps track of requesting and receiving PR2 data from a single sensor.
/// The point is not to block the main loop with delays between commands.
/// It uses common timer pr2_delay_timer, which must be global.
class Teros31Reader{
  private:
    char _address;
    SDI12Comm * _sdi12_comm;

    float rec_values[3];
    uint8_t rec_n_values = 0;

  public:
    Teros31Data data;
    bool finished = false;
    bool requested = false;
    uint8_t n_tryouts = 0;

    Teros31Reader(SDI12Comm* sdi12_comm, char address);
    bool TryRequest();
    void TryRead();
    void Reset();
};


Teros31Reader::Teros31Reader(SDI12Comm* sdi12_comm, char address)
  :_address(address), _sdi12_comm(sdi12_comm)
{
  Reset();
}

bool Teros31Reader::TryRequest()
{
  if(!sdi12_delay_timer.running)
  {
    bool res = false;
    char* msg = _sdi12_comm->measureRequest("C", _address, &res);
    if(res){
      sdi12_delay_timer.reset();
      n_tryouts = 0;  // reset
      requested = true;
      // Logger::printf(Logger::INFO, "sdi12[a%c]: ", );
    }
    else{
      n_tryouts++;
    }
    return res;
  }
  return true;
}

void Teros31Reader::TryRead()
{
  if(sdi12_delay_timer.after())
  {
    char* res = _sdi12_comm->measureRead(_address, rec_values, &rec_n_values);
    // _sdi12_comm.print_values("field", rec_values, rec_n_values);
    if(res != nullptr)
    {
      if(rec_n_values>0)
        data.pressure = rec_values[0];
      if(rec_n_values>1)
        data.temperature = rec_values[1];
      finished = true;
      n_tryouts = 0;
    }
    else
      n_tryouts++;
  }
}

void Teros31Reader::Reset()
{
  n_tryouts = 0;
  requested = false;
  finished = false;
  data = Teros31Data();
}

#endif // TEROS31_READER_H_
