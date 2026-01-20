#ifndef WATER_HEIGHT_SENSOR_H_
#define WATER_HEIGHT_SENSOR_H_

#include "ESP32AnalogRead.h"
#include "stdlib.h"

class WaterHeightSensor
{
  private:
    ESP32AnalogRead _adc;
    const uint8_t _pin;
    // _minWindow <= _minVoltate < maxVoltage <= _maxWindow
    const float _minVoltage;  // [V]
    const float _maxVoltage;  // [V]
    const float _minHeight;   // [mm]
    const float _maxHeight;   // [mm]
    float _k,_q;

    // possibly set operation window for the ultrasonic sensor S18U
    float _minWindow;   // [V]
    float _maxWindow;   // [V]

  public:
  /**
   * @brief Construct a new Water Height Sensor object
   *
   * @param pin
   * @param minH  minimal height (interpolation)
   * @param maxH  maximal height
   * @param minV  minimal voltage
   * @param maxV  maximal voltage
   */
    WaterHeightSensor(uint8_t pin, float minH, float maxH, float minV, float maxV);
    void begin();
    /**
     * @brief Set the Voltage Window (defined property of S18U sensor).
     *
     * @param minW minimal voltage (defined by TEACH mode)
     * @param maxW maximal voltage (defined by TEACH mode)
     */
    void setWindow(float minW, float maxW);
    float read(float* voltage);
};

WaterHeightSensor::WaterHeightSensor(uint8_t pin, float minH, float maxH, float minV, float maxV)
: _pin(pin), _minHeight(minH), _maxHeight(maxH), _minVoltage(minV), _maxVoltage(maxV)
{
  float rangeVoltage = _maxVoltage - _minVoltage;
  float rangeHeight = _maxHeight - _minHeight;
  _k = - rangeHeight / rangeVoltage;
  _q = _maxHeight - _k*_minVoltage;

  _minWindow = _minVoltage;
  _maxWindow = _maxVoltage;
}

void WaterHeightSensor::begin()
{
  _adc.attach(_pin);
}

void WaterHeightSensor::setWindow(float minW, float maxW)
{
  _minWindow = minW;
  _maxWindow = maxW;
}

float WaterHeightSensor::read(float* volt)
{
  float voltage = _adc.readVoltage();
  *volt = voltage;

  // check window range
  float nan = std::numeric_limits<float>::quiet_NaN();
  if(voltage <= _minWindow || voltage >= _maxWindow)
    return nan;

  // check interpolation range
  if(voltage <= _minVoltage)
    return _minHeight;
  else if(voltage >= _maxVoltage)
    return _maxHeight;

  // voltage in range:
  float height = _k * voltage + _q; //ratio
  return height;
}

#endif // WATER_HEIGHT_SENSOR_H_
