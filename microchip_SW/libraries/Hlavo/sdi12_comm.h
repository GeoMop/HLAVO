#ifndef SDI12_COMM_H_
#define SDI12_COMM_H_

#include "common.h"
#include <Every.h>
#include <SDI12.h>
#include "Logger.h"
#include "stdlib.h"

#ifndef defined(ESP32) || defined(ESP8266)
  // Fix no strof for Arduino
  #define strtof(A, B) strtod(A, B)
#endif

// Global timer for PR2 request-read delay.
Timer sdi12_delay_timer(2200, false);

/// @brief SDI12 wrapper class.
/// Uses some key approaches from ESP32_SDI12, reads byte by byte to
/// resolves invalid characters at beginning of messages.
class SDI12Comm
{
  private:
    // static const uint16_t _measure_delay = 2200; // waiting between measure request and data request
    // static const uint8_t n_bytes_per_val = 7;
    static const uint8_t max_msg_length = 100;
    static const int8_t extra_wake_delay = 0;
    // static const char* crlf;

    uint8_t _verbose = 0;
    SDI12 _SDI12;

    uint8_t _n_expected;
    char _msg_buf[max_msg_length];

  public:
    SDI12Comm(int8_t dataPin, uint8_t verbose=0);

    void begin();

    // String requestAndReadData(String command, bool trim = false);
    // String measureConcurrent(String measure_command, char address);

    char* requestAndReadData(const char* command, uint8_t* n_bytes);
    char* measureRequest(String measure_command, char address, bool *result);
    char* measureRead(char address, float* values, uint8_t* n_values);
    char* measureRequestAndRead(String measure_command,char address, float* values, uint8_t* n_values);
    void print_values(String field_name, float* values, uint8_t n_values);

  private:
    void print_response(String cmd, String response);
    void print_response(String cmd, const char* response);
    char* findFirstDigit(char* str, uint8_t n_bytes);
};

// const char * PR2Comm::crlf = "\r\n";

SDI12Comm::SDI12Comm(int8_t dataPin, uint8_t verbose)
  : _verbose(verbose), _SDI12(dataPin)
{
}

void SDI12Comm::begin()
{
  // Serial.print("SDI12 datapin: ");
  // Serial.println(_SDI12.getDataPin());
  // Serial.println(_verbose);
  _SDI12.begin();
}


// String PR2Comm::requestAndReadData(String command, bool trim = false) {

  // return String(requestAndReadData(command.c_str()));

  // // delay(300);                   // Wait for response to be ready
  // // _SDI12.clearBuffer();
  // _SDI12.sendCommand(command); // Send the SDI-12 command
  // delay(50);                   // Wait for response to be ready

  // String sensorResponse = "";
  // //Read the response from the sensor
  // while (_SDI12.available()) { // Check if there is data available to read
  //   char c = _SDI12.read();    // Read a single character
  //   // if (c != -1 && c != 0x00 && c != 0x7F) {              // Check if the character is valid
  //   if (c != -1) {              // Check if the character is valid
  //     sensorResponse += c;      // Append the character to the response string
  //     if(_verbose > 1){
  //       Serial.print(c, HEX); Serial.print(" ");
  //     }
  //   }
  //   delay(10);  // otherwise it would leave some chars to next message...
  // }

  // if(_verbose > 1)
  //   Serial.println("");

  // if (trim)
  //   sensorResponse.trim();  // remove CRLF at the end

  // if(_verbose > 0)
  // {
  //   // if printf not available (Arduino)
  //   // char string_buffer[128]; // Buffer to hold the formatted string
  //   // snprintf(string_buffer, sizeof(string_buffer), "command %s: %s", command.c_str(), sensorResponse.c_str());
  //   // Serial.println(string_buffer);
  //   print_response(command, sensorResponse);
  // }

  // return sensorResponse;
// }

// String PR2Comm::measureConcurrent(String measure_command, char address)
// {
//     measure_command = String(address) + measure_command + "!";
//     String measureResponse = requestAndReadData(measure_command, true);  // Command to take a measurement
//     // print_response(measure_command, measureResponse);

//     // last value is number of measured values
//     uint8_t n_values = atoi(measureResponse.end()-1);
//     Serial.println(n_values);
//     uint8_t total_length = n_values*n_bytes_per_val;
//     Serial.println(total_length);

//     delay(_measure_delay);

//     // for C commands
//     // max message length 75 B
//     uint8_t position = 0;
//     String data_command = String(address) + "D" + String(position) + "!";
//     String sensorResponse = requestAndReadData(data_command, true);

//     // print_response(data_command, sensorResponse);

//     // for M commands - multiple D commands necessary
//     // max message length 35 B
//     // Position in the data command request (multiple calls may be needed to
//     // get all values from a sensor).
//     // String sensorResponse = "";
//     // uint8_t position = 0;
//     // const uint8_t max_values = 10; // aD0! ... aD9! range for PR2
//     // while (position < max_values) {
//     //     // Request data as it should be ready to be read now
//     //     String data_command = String(address) + "D" + String(position) + "!";
//     //     // Serial.println(data_command);
//     //     sensorResponse += requestAndReadData(data_command);

//     //     // Increment the position in the data command to get more measurements
//     //     // until all values hav been received
//     //     position++;
//     //     delay(20);
//     // }
//     // 30 2B 30 2E 39 39 34 35 2B 31 2E 30 30 34 35 2B 31 2E 30 32 32 39 2B 30 2E 39 38 34 38 2B 30 2E 39 38 35 36 2B 30 2E 39 38 34 38 D A
//     // 30 2B 30 2E 39 39 34 33 2B 31 2E 30 30 35 2B 31 2E 30 32 33 31 2B 30 2E 39 38 34 38 2B 30 2E 39 38 35 33 2B 30 2E 39 38 35 D A
//     // 0+0.9945+1.0045+1.0229+0.9848+0.9856+0.9848
//     // 0+0.9943+1.005+1.0231+0.9848+0.9853+0.985
//     Serial.println(sensorResponse.length());
//     Serial.println(sensorResponse.length() - total_length);
//     return sensorResponse.substring(sensorResponse.length() - total_length);
// }



char* SDI12Comm::requestAndReadData(const char* command, uint8_t* n_bytes) {

  for(int i=0; i<max_msg_length; i++)
    _msg_buf[i] = 0;

  _SDI12.clearBuffer();
  _SDI12.clearWriteError();
  delay(20);

  _SDI12.sendCommand(command, extra_wake_delay); // Send the SDI-12 command
  delay(20);                   // Wait for response to be ready
  // Logger::printf(Logger::INFO, "Command: '%s'\n", command);

  uint8_t counter = 0;
  //Read the response from the sensor
  while (_SDI12.available()) { // Check if there is data available to read

    char c = _SDI12.read();    // Read a single character
    _msg_buf[counter] = c;      // Append the character to the response string
    counter++;

    if(_verbose > 0){
      Serial.print(c, HEX); Serial.print(" ");
    }
    if(counter >= max_msg_length)
    {
      if(_verbose > 1)
        Logger::print("sdi12::requestAndReadData Max length reached!", Logger::ERROR);
      break;
    }
    delay(8);  // otherwise it would leave some chars to next message...
  }
  _SDI12.forceHold();

  *n_bytes = counter;

  if(counter>0)
  {
    if(_verbose > 0)
      Serial.println("");
    // Logger::printHex(_msg_buf, counter);
    Logger::printf(Logger::INFO, "sdi12[%s]: %s", command, _msg_buf);
  }
  else{
    if(_verbose > 1)
      Logger::printf(Logger::ERROR, "sdi12 [%s] - no response!\n", command);
  }

  if(_verbose > 1)
  {
    // if printf not available (Arduino)
    // char string_buffer[128]; // Buffer to hold the formatted string
    // snprintf(string_buffer, sizeof(string_buffer), "command %s: %s", command.c_str(), sensorResponse.c_str());
    // Serial.println(string_buffer);
    print_response(command, _msg_buf);
  }

  return _msg_buf;
}


char* SDI12Comm::measureRequest(String measure_command, char address, bool *result)
{
  uint8_t n_bytes = 0;
  measure_command = String(address) + measure_command + "!";
  requestAndReadData(measure_command.c_str(), &n_bytes);  // Command to take a measurement

  if(n_bytes != 8)
  {
    if(n_bytes>0 && _verbose > 1)
      Logger::printf(Logger::ERROR, "sdi12[%s] - no valid response received!\n", measure_command.c_str());
    *result = false;
    return nullptr;
  }

  // for(int i=0; i<n_bytes; i++)
    // Serial.print(_msg_buf[i], HEX);
    // Serial.printf("%02X ",_msg_buf[i]);
  // Serial.println();
  // char* msg_start = findFirstDigit(_msg_buf, n_bytes);
  // uint8_t raddress = *msg_start - '0';//strtof(msg_start, &msg_ptr);

  // is it a valid message?
  bool res = true;
  res = res && _msg_buf[n_bytes-1] == '\n';  // 0A
  res = res && _msg_buf[n_bytes-2] == '\r';  // 0D
  char raddress = _msg_buf[0];
  res = res && raddress == address;           // returns the requested address
  uint8_t delay_time = _msg_buf[3]-'0';       // delay time in seconds
  res = res && (0 < delay_time) && (delay_time < 10);  // delay time between 1-9
  uint8_t n_vals = _msg_buf[n_bytes-3]-'0';       // n data
  res = res && (0 < delay_time) && (delay_time < 10);  // n data between 1-9

  *result = res;

  if(res)
  {
    sdi12_delay_timer.interval = delay_time*1000;
    _n_expected = n_vals;
    if(_verbose>1)
      hlavo::SerialPrintf(30, "_n_expected: %d, delay: %ds\n", _n_expected, delay_time);
  }
  else
  {
    if(_verbose>1)
      Logger::print("Invalid message received.", Logger::MessageType::ERROR);
    _n_expected = 0;
    return _msg_buf;
  }
  return _msg_buf;
}

char* SDI12Comm::measureRead(char address, float* values, uint8_t* n_values)
{
  uint8_t n_bytes = 0;
  // for C commands
  // max message length 75 B
  uint8_t position = 0;
  String data_command = String(address) + "D" + String(position) + "!";
  requestAndReadData(data_command.c_str(), &n_bytes);  // Command to read measurement data

  if(n_bytes <= 5)
  {
    if(_verbose > 1)
      Logger::printf(Logger::ERROR, "sdi12[%s] - no valid response received!\n", data_command.c_str());
    *n_values = 0;
    return nullptr;
  }

  // char* msg_start = findFirstDigit(_msg_buf, n_bytes);
  // if(_verbose>0)
    // hlavo::SerialPrintf(120, "cleared msg: %s\n", msg_start);

  // is it a valid message?
  bool res = true;
  res = res && _msg_buf[n_bytes-1] == '\n';  // 0A
  res = res && _msg_buf[n_bytes-2] == '\r';  // 0D
  char raddress = _msg_buf[0];
  res = res && raddress == address;           // returns the requested address

  if(! res)
  {
    if(_verbose > 1)
      Logger::print("Invalid message received.", Logger::MessageType::ERROR);
    *n_values = 0;
    return _msg_buf;
  }

  uint8_t parsed_values = 0; // Number of values successfully parsed
  char* msg_ptr = _msg_buf + 1; // skip leading address
  // Extracts the device address and stores a ptr to the rest of the
  // message buffer for use below (to extract values only)
  // strtof(msg_start, &msg_ptr);

  char* next_msg_ptr;
  float value;
  // Extract the values from the message buffer and put into user
  // supplied buffer
  // Serial.printf("_n_expected: %d\n", _n_expected);
  for (size_t i = 0; i < _n_expected; i++) {
      value = strtof(msg_ptr, &next_msg_ptr);
      if(msg_ptr == next_msg_ptr){
          // while(*msg_ptr != '+' || *msg_ptr != '-' || msg_ptr != _msg_buf+nbytes)
          //   msg_ptr = msg_ptr + 1;
          break;
      }
      // Serial.printf("Value: %f\n", value);
      values[parsed_values++] = value;
      msg_ptr = next_msg_ptr;
  }

  *n_values = parsed_values;
  return _msg_buf;
}

char* SDI12Comm::measureRequestAndRead(String measure_command,char address, float* values, uint8_t* n_values)
{
  bool res = false;
  measureRequest(measure_command, address, &res);

  delay(sdi12_delay_timer.interval);

  return measureRead(address, values, n_values);
}

void SDI12Comm::print_values(String field_name, float* values, uint8_t n_values)
{
  // hlavo::SerialPrintf(100, "%-25s", (field_name + ':').c_str());
  for(int i=0; i<n_values; i++)
    hlavo::SerialPrintf(10, "%.4f  ", values[i]);
  Serial.println();
}


void SDI12Comm::print_response(String cmd, String response)
{
  hlavo::SerialPrintf(20, "command %s: %s\n", cmd.c_str(), response.c_str());
}

void SDI12Comm::print_response(String cmd, const char* response)
{
  hlavo::SerialPrintf(150, "command %s: %s\n", cmd.c_str(), response);
}

char* SDI12Comm::findFirstDigit(char* str, uint8_t n_bytes) {
  // Loop through each character until we hit the string's null terminator
  for(int i=0; i<n_bytes; i++) {
      // Check if the current character is a digit
      if (*str >= '0' && *str <= '9') {
          return str;  // Return the pointer to the current character
      }
      if(_verbose >0)
        hlavo::SerialPrintf(30,"skipping %X\n", *str);
      str++;  // Move to the next character
  }
  return nullptr;  // Return nullptr if no digit is found
}

#endif // SDI12_COMM_H_
