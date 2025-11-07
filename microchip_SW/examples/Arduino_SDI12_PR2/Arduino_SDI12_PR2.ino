
#include <Arduino.h>
#include <SDI12.h>

#define DEBUG 1

#define SERIAL_BAUD 115200 /*!< The baud rate for the output serial port */
#define DATA_PIN 8         /*!< The pin of the SDI-12 data bus */
#define POWER_PIN 22       /*!< The sensor power pin (or -1 if not switching power) */


/** Define the SDI-12 bus */
SDI12 mySDI12(DATA_PIN);


void setup() {
  Serial.begin(SERIAL_BAUD);
  while (!Serial)
    ;

  Serial.println("Opening SDI-12 bus...");
  mySDI12.begin();
  delay(500);  // allow things to settle

  // Power the sensors;
  if (POWER_PIN > 0) {
    Serial.println("Powering up sensors...");
    pinMode(POWER_PIN, OUTPUT);
    digitalWrite(POWER_PIN, HIGH);
    delay(500);//(200);
  }
}


String requestAndReadData(String command, bool trim = false) {
  mySDI12.sendCommand(command); // Send the SDI-12 command
  delay(300);                   // Wait for response to be ready
  
  String sensorResponse = "";
  //Read the response from the sensor
  while (mySDI12.available()) { // Check if there is data available to read
    char c = mySDI12.read();    // Read a single character
    if (c != -1) {              // Check if the character is valid
      sensorResponse += c;      // Append the character to the response string
      #if DEBUG
        Serial.print(c, HEX); Serial.print(" ");
      #endif
    }
    delay(20);  // otherwise it would leave some chars to next message...
  }
  #if DEBUG
    Serial.println("");
  #endif
// >> 30 31 33 44 65 6C 74 61 2D 54 20 50 52 32 53 44 49 31 2E 31 50 52 32 2F 36 2D 30 34 35 30 36 30 D A command ?I!: 013Delta-T PR2SDI1.1PR2/6-045060
// >> 30 30 30 32 36 D A command 0M!: 00026
// >> 30 D A 30 2B 30 2E 39 38 32 2B 30 2E 39 38 2B 31 2E 30 30 34 38 2B 30 2E 39 38 35 34 2B 30 2E 39 38 37 35 D command 0D0!: 00+0.982+0.98+1.0048+0.9854+0.9875
// >> A 30 2B 30 2E 39 38 30 32 D A command 0D1!: 0+0.9802
// >> DATA: 00+0.982+0.98+1.0048+0.9854+0.98750+0.9802

  // command 0D0! returns <address><CRLF><data><CRLF> => readStringUntil would need to called twice
  // String sensorResponse = mySDI12.readStringUntil('\n');
  
   // Replace \r and \n with empty strings
  // sensorResponse.replace("\r", "");
  // sensorResponse.replace("\n", "");
  if (trim)
    sensorResponse.trim();  // remove CRLF at the end

  #if DEBUG
    char string_buffer[128]; // Buffer to hold the formatted string
    snprintf(string_buffer, sizeof(string_buffer), "command %s: %s", command.c_str(), sensorResponse.c_str());
    Serial.println(string_buffer);
  #endif

  return sensorResponse;
}

// replacement of strtof() for arduino
// float customStrtof(const char* str, char** str_end) {
//   float result = 0.0;
//   while (*str == ' ' || *str == '\t' || *str == '\r' || *str == '\n') { // skip any leading whitespace
//     str++;
//   }

//   // // Check for and skip over leading "00"
//   // if (*str == '0' && *(str + 1) == '0') {
//   //   str += 2;
//   // }

//   // Find the start of the number (skip over any signs that are not part of a number)
//   const char* start = str;
//   if ((*start >= '0' && *start <= '9') || *start == '+' || *start == '-') {
//     start++;
//   }

//   // Find the end of the number
//   while ((*start >= '0' && *start <= '9') || *start == '.' || *start == 'e' || *start == 'E' || (*start == '+' || *start == '-') && (start[-1] == 'e' || start[-1] == 'E')) {
//     start++;
//   }

//   // Temporary buffer to hold the number for conversion
//   char numberBuffer[32];
//   int numLength = start - str;
//   if (numLength > 31) numLength = 31; // Avoid buffer overflow
//   strncpy(numberBuffer, str, numLength);
//   numberBuffer[numLength] = '\0';

//   result = atof(numberBuffer); // Convert the extracted number to a float

//   if (str_end != NULL) { // Provide the next start point if required
//     *str_end = (char*)start;
//   }

//   return result;
// }

// /**
//  * Ask SDI-12 sensor to take a measurement and return all values.
//  *
//  * @breif Combines the requestMeasure() and requestData() commands to ask a
//  * sensor to take any measurements and wait for a response then return all
//  * these data as floats to the user. This will handle multiple data commands
//  * and only return when all values have been received.
//  *
//  * @code
//  * static float values[10];
//  * static uint8_t n_received;
//  * ESP32_SDI12::Status res = sdi12.measure(address, values,
//  *                                          sizeof(values), &n_received);
//  *
//  * @param address Sensor SDI-12 address.
//  * @param values User supplied buffer to hold returned measurement values.
//  * @param max_values Number of values in user supplied buffer.
//  * @param num_returned_values The number of values read back from the sensor.
//  * @return Status code (ESP32_SDI12::Status).
//  */
// void measure(String measure_command,
//         uint8_t address,
//         float* values,
//         size_t max_values,
//         uint8_t* num_returned_values)
// {
//     measure_command = String(address) + measure_command + "!";
//     String measureResponse = requestAndReadData(measure_command);  // Command to take a measurement
//     // Serial.println(measureResponse);
//     delay(2000);

//     // Tell the caller they did not provide enough space for the measurements.
//     // if (measure.n_values >= max_values) {
//     //     return SDI12_BUF_OVERFLOW;
//     // }

//     // Serial.printf("SDI12_BUF_OVERFLOW: %d %d\n", measure.n_values, max_values);
//     // const size_t max_bufsize = 128;
//     // char msg_buf[max_bufsize] = {0};

//     uint8_t parsed_values = 0; // Number of values successfully parsed
//     // Position in the data command request (multiple calls may be needed to
//     // get all values from a sensor).
//     uint8_t position = 0;
//     while (parsed_values < max_values) {
//         // Serial.println("parsed_values: %d\n", parsed_values);
//         // Request data as it should be ready to be read now
//         String data_command = String(address) + "D" + String(position) + "!";
//         // Serial.println(data_command);
//         String sensorResponse = requestAndReadData(data_command);
//         // mySDI12.sendCommand(command); // Send the SDI-12 command
//         // delay(300);                   // Wait for response to be ready

//         //delay(500);
//         // Serial.println("requestData done");

//         // String sensorResponse = "";
//         // while (mySDI12.available()) { // Check if there is data available to read
//         //   char c = mySDI12.read();    // Read a single character
//         //   if (c != -1) {              // Check if the character is valid
//         //     sensorResponse += c;      // Append the character to the response string
//         //     // Serial.print(c, HEX); Serial.print(" ");
//         //   }
//         // }

//         // Serial.print(sensorResponse);
//         char* msg_ptr;
//         char* msg_buf = sensorResponse.c_str();
//         // Extracts the device address and stores a ptr to the rest of the
//         // message buffer for use below (to extract values only)
//         float val;
//         val = customStrtof(msg_buf, &msg_ptr);
//         msg_buf = msg_ptr;
//         Serial.println(val); Serial.println(msg_ptr);
        
//         val = customStrtof(msg_buf, &msg_ptr);
//         msg_buf = msg_ptr;
//         Serial.println(val); Serial.println(msg_ptr);

//         // val = customStrtof(msg_buf, &msg_ptr);
//         // Serial.print(val); Serial.println(msg_ptr);

//         char* next_msg_ptr;
//         float value;
//         // Extract the values from the message buffer and put into user
//         // supplied buffer
//         for (size_t i = 0; i < max_values; i++) {
//             value = customStrtof(msg_ptr, &next_msg_ptr);
//             if(msg_ptr == next_msg_ptr){
//                 break;
//             }
            
//             Serial.println(value, 4);
            
//             values[parsed_values++] = value;
//             msg_ptr = next_msg_ptr;
//         }

//         // Increment the position in the data command to get more measurements
//         // until all values hav been received
//         position++;
//     }

//     // Assign the number for returned values to the user provided variable
//     // Skip of num_returned_values is a nullptr
//     // if (num_returned_values) {
//     //     *num_returned_values = measure.n_values;
//     // }
//     *num_returned_values = parsed_values;
// }

String measureString(String measure_command, uint8_t address)
{
    measure_command = String(address) + measure_command + "!";
    String measureResponse = requestAndReadData(measure_command);  // Command to take a measurement
    // Serial.println(measureResponse);
    delay(2000);

    // for C commands
    uint8_t position = 0;
    String data_command = String(address) + "D" + String(position) + "!";
    String sensorResponse = requestAndReadData(data_command);

    // for M commands - multiple D commands
    // Position in the data command request (multiple calls may be needed to
    // get all values from a sensor).
    // String sensorResponse = "";
    // uint8_t position = 0;
    // const uint8_t max_values = 10; // aD0! ... aD9! range for PR2
    // while (position < max_values) {
    //     // Request data as it should be ready to be read now
    //     String data_command = String(address) + "D" + String(position) + "!";
    //     // Serial.println(data_command);
    //     sensorResponse += requestAndReadData(data_command);

    //     // Increment the position in the data command to get more measurements
    //     // until all values hav been received
    //     position++;
    //     delay(20);
    // }
    return sensorResponse;
}

bool human_print = true;
void loop() {
  
  String sensorResponse = "";

  // sensorResponse = requestAndReadData("?I!", true);  // Command to get sensor info
  // sensorResponse = requestAndReadData("?I!", false);  // Command to get sensor info
  //Serial.println(sensorResponse);

  // float values[6];
  // size_t max_values = 10;
  // uint8_t num_values = 0;
  // measure("M",0,values,max_values,&num_values);


  uint8_t addr = 0;
  sensorResponse = requestAndReadData(String(addr) + "I!", false);  // Command to get sensor info
  if(human_print) Serial.print("info:    ");
  Serial.println(sensorResponse);

  sensorResponse = measureString("C",addr);
  //sensorResponse = sensorResponse.substring(3); // first 3 characters are <address><CR><LF> => remove
  sensorResponse = sensorResponse.substring(1); // first 1 characters is <address> => remove
  if(human_print) Serial.print("permitivity:    ");
  Serial.println(sensorResponse);

  // sensorResponse = measureString("C1",addr);
  // sensorResponse = sensorResponse.substring(1); // first 1 characters is <address> => remove
  // if(human_print) Serial.print("soil moisture:  ");
  // Serial.println(sensorResponse);

  // sensorResponse = measureString("C8",addr);
  // sensorResponse = sensorResponse.substring(1); // first 1 characters is <address> => remove
  // if(human_print) Serial.print("millivolts:     ");
  // Serial.println(sensorResponse);

  // sensorResponse = measureString("C9",addr);
  // sensorResponse = sensorResponse.substring(1); // first 1 characters is <address> => remove
  // if(human_print) Serial.print("raw ADC:        ");
  // Serial.println(sensorResponse);

  // char string_buffer[128]; // Buffer to hold the formatted string
  // snprintf(string_buffer, sizeof(string_buffer), "DATA: %s", sensorResponse.c_str());
  // Serial.println(string_buffer);
}