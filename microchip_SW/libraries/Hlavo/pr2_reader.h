#ifndef PR2_READER_H_
#define PR2_READER_H_

#include "sdi12_comm.h"
#include "pr2_data.h"

/// @brief Class that keeps track of requesting and receiving PR2 data from a single sensor.
/// The point is not to block the main loop with delays between commands.
/// It uses common timer pr2_delay_timer, which must be global.
class PR2Reader{
  private:
    char _address;
    SDI12Comm* _sdi12_comm;

    static const uint8_t _n_fields = 3;
    const char* _list_of_commands[_n_fields] = {"C", "C1", "C9"};
    /**
     * @brief 
     * C1: Soil moisture measurement in Theta (m3.m-3) with Mineral soil type calibration
     * C3: Soil moisture measurement as percentage volumetric (%) with Mineral soil type calibration
     */

    uint8_t icmd = 0;

    float rec_values[10];
    uint8_t rec_n_values = 0;

  public:
    PR2Data data;
    bool finished = false;

    PR2Reader(SDI12Comm* sdi12_comm, char address);
    bool TryRequest();
    void TryRead();
    void Reset();
};


PR2Reader::PR2Reader(SDI12Comm* sdi12_comm, char address)
  :_address(address), _sdi12_comm(sdi12_comm)
{
  Reset();
}

bool PR2Reader::TryRequest()
{
  // Logger::printf(Logger::INFO, "PR2Reader::TryRequest %d %d", sdi12_delay_timer.interval, sdi12_delay_timer.running);
  if( ! (sdi12_delay_timer.running))
  {
    bool res = false;
    char* msg = _sdi12_comm->measureRequest(_list_of_commands[icmd], _address, &res);
    if(res)
    {
      sdi12_delay_timer.reset();
      // Logger::printf(Logger::INFO, "PR2Reader::TryRequest TRUE %d %d", sdi12_delay_timer.interval, sdi12_delay_timer.running);
    }
    return res;
  }
  return true;
}

void PR2Reader::TryRead()
{
  // Logger::printf(Logger::INFO, "PR2Reader::TryRead %d %d", sdi12_delay_timer.interval, sdi12_delay_timer.running);
  if(sdi12_delay_timer.after())
  {
    // Logger::print("PR2Reader::TryRead timer finished");
    char* res = _sdi12_comm->measureRead(_address, rec_values, &rec_n_values);
    // _pr2_comm.print_values("field", rec_values, rec_n_values);
    if(res != nullptr)
    {
      switch(icmd)
      {
        case 0: data.setPermitivity(rec_values, rec_n_values); break;
        case 1: data.setSoilMoisture(rec_values, rec_n_values); break;
        case 2: if(rec_n_values > 0)
                  data.setRaw_ADC(&rec_values[1], rec_n_values-1);
                break;
      }
    }
    icmd++;

    if(icmd == _n_fields)
    {
      icmd = 0;
      finished = true;
    }
  }
}

void PR2Reader::Reset()
{
  icmd = 0;
  finished = false;
  data = PR2Data();
}

#endif // PR2_READER_H_
