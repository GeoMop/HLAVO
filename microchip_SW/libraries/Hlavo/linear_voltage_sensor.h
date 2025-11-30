#ifndef LINEAR_VOLTAGE_SENSOR_H_
#define LINEAR_VOLTAGE_SENSOR_H_

#include "ESP32AnalogRead.h"
#include "stdlib.h"

class LinearVoltageSensor
{
  private:
    ESP32AnalogRead _adc;
    const uint8_t _pin;
    // _aWindow <= _aVoltage < bVoltage <= _bWindow
    const float _aVoltage;  // [V]
    const float _bVoltage;  // [V]
    const float _aValue;    // f(_aVoltage)
    const float _bValue;    // f(_bVoltage)
    // linearization constants
    float _k,_q;

    // possibly set operation window (e.g. for the ultrasonic sensor S18U)
    float _aWindow;   // [V]
    float _bWindow;   // [V]

  public:
  /**
   * @brief Construct a new Linerar Voltage Sensor object
   *
   * @param pin
   * @param aVolt  minimal voltage
   * @param bVolt  maximal voltage
   * @param aVal  minimal value
   * @param bVal  maximal value
   */
    LinearVoltageSensor(uint8_t pin, float aVolt, float bVolt, float aVal, float bVal);
    void begin();
    /**
     * @brief Set the Voltage Window (defined property of S18U sensor).
     *
     * @param aWindow minimal voltage (defined by TEACH mode of S18U)
     * @param bWindow maximal voltage (defined by TEACH mode of S18U)
     */
    void setWindow(float aWindow, float bWindow);
    float read(float* voltage);
};

LinearVoltageSensor::LinearVoltageSensor(uint8_t pin, float aVolt, float bVolt, float aVal, float bVal)
: _pin(pin), _aVoltage(aVolt), _bVoltage(bVolt), _aValue(aVal), _bValue(bVal)
{
  float rangeVoltage = _bVoltage - _aVoltage;
  float rangeValue = _bValue - _aValue;
  _k = rangeValue / rangeVoltage;
  _q = _aValue - _k*_aVoltage;

  _aWindow = _aVoltage;
  _bWindow = _bVoltage;
}

void LinearVoltageSensor::begin()
{
  _adc.attach(_pin);
}

void LinearVoltageSensor::setWindow(float aWindow, float bWindow)
{
  _aWindow = aWindow;
  _bWindow = bWindow;
}

float LinearVoltageSensor::read(float* volt)
{
  float voltage = _adc.readVoltage();
  *volt = voltage;

  // check window range
  float nan = std::numeric_limits<float>::quiet_NaN();
  if(voltage <= _aWindow || voltage >= _bWindow)
    return nan;

  // check interpolation range
  if(voltage <= _aVoltage)
    return _aValue;
  else if(voltage >= _bVoltage)
    return _bValue;

  // voltage in range:
  float value = _k * voltage + _q;
  return value;
}

#endif // LINEAR_VOLTAGE_SENSOR_H_
