#include "WeatherMeters.h"

class WeatherStation {
  private:
    const uint8_t _windvane_pin;
    const uint8_t _anemometer_pin;
    const uint8_t _raingauge_pin;

    hw_timer_t * _timer;
    volatile SemaphoreHandle_t _timerSemaphore;
    volatile bool _do_update = false;
    portMUX_TYPE _timerMux = portMUX_INITIALIZER_UNLOCKED;

    WeatherMeters <4> meters; // filter last 4 directions

    // callback to detect data read finish
    static volatile bool _got_data;
    static void gotDataHandler(void) { _got_data = true; }

  public:

    WeatherStation(uint8_t windvane_pin, uint8_t anemometer_pin, uint8_t raingauge_pin, uint16_t period);
    void setup(WeaterMetersCallback intAnemometer, WeaterMetersCallback intRaingauge, WeaterMetersCallback intPeriod);

    // Interrupt function for anemometer tick.
    void intAnemometer() {
      meters.intAnemometer();
    }

    // Interrupt function for raingauge tick.
    void intRaingauge() {
      meters.intRaingauge();
    }

    // Interrupt function for period timer.
    void intTimer() {
      xSemaphoreGiveFromISR(_timerSemaphore, NULL);
      _do_update = true;
    }

    /** To be called inside loop():
     * weather.update();
     * if (weather.gotData()) {
     *   ...
     *   weather.resetGotData();
     * }
     */
    void update()
    {
      if(_do_update){
        meters.timer();
        _do_update = false;
	    }
    }

    bool gotData(){
      return _got_data;
    }

    void resetGotData(){
      _got_data = false;
    }


    float getDirection() {
      return meters.getDir();
    }

    float getSpeed() {
      return meters.getSpeed();
    }

    unsigned int getSpeedTicks() {
      return meters.getSpeedTicks();
    }

    unsigned int getRainTicks() {
      return meters.getRainTicks();
    }

    unsigned int getDirAdcValue() {
      return meters.getDirAdcValue();
    }

    float getRain_mm() {
      return meters.getRain_mm();
    }

    float getRain_ml() {
      return meters.getRain_ml();
    }
};


WeatherStation::WeatherStation(uint8_t windvane_pin, uint8_t anemometer_pin, uint8_t raingauge_pin, uint16_t period)
  : _windvane_pin(windvane_pin),
  _anemometer_pin(anemometer_pin),
  _raingauge_pin(raingauge_pin),
  _timer(NULL),
  meters(_windvane_pin, period) // refresh data every * sec
{
}

void WeatherStation::setup(WeaterMetersCallback intAnemometer,
  WeaterMetersCallback intRaingauge,
  WeaterMetersCallback intPeriod)
{
  pinMode(_windvane_pin, ANALOG);
  pinMode(_raingauge_pin, INPUT_PULLUP);  // Set GPIO as input with pull-up (like adding 10k resitor)
  pinMode(_anemometer_pin, INPUT_PULLUP);  // Set GPIO as input with pull-up (like adding 10k resitor)
  attachInterrupt(digitalPinToInterrupt(_anemometer_pin), intAnemometer, CHANGE);
  attachInterrupt(digitalPinToInterrupt(_raingauge_pin), intRaingauge, CHANGE);

  meters.attach(gotDataHandler);

  _timerSemaphore = xSemaphoreCreateBinary();
  _timer = timerBegin(0, 80, true);
  timerAttachInterrupt(_timer, intPeriod, true);
  timerAlarmWrite(_timer, 1000000, true);
  timerAlarmEnable(_timer);

  meters.reset();  // in case we got already some interrupts
}


volatile bool WeatherStation::_got_data = false;
