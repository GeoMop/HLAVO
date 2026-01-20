
#include <Arduino.h>
#include <SoftwareSerial.h>

#define DEBUG 1

#define SERIAL_BAUD 115200 /*!< The baud rate for the output serial port */
#define DATA_PIN 8
#define POWER_PIN 22       /*!< The sensor power pin (or -1 if not switching power) */


SoftwareSerial mySerial(DATA_PIN, DATA_PIN); // RX, TX
// SoftwareSerial mySerial(DATA_PIN, DATA_PIN, true); // RX, TX, inverse_logic

void setup() {
  Serial.begin(SERIAL_BAUD);
  while (!Serial)
    ;

  Serial.println("Opening serial bus...");
  // set the data rate for the SoftwareSerial port
  mySerial.begin(1200);
  delay(500);  // allow things to settle

  // // Power the sensors;
  // if (POWER_PIN > 0) {
  //   Serial.println("Powering up sensors...");
  //   pinMode(POWER_PIN, OUTPUT);
  //   digitalWrite(POWER_PIN, HIGH);
  //   delay(500);//(200);
  // }
  Serial.println("Setup finished.");
}


String ReadData() {
  String sensorResponse = "";
  //Read the response from the sensor
  while (mySerial.available()) { // Check if there is data available to read
    Serial.println("available");
    char c = mySerial.read();    // Read a single character
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

  return sensorResponse;
}



void printStringAsHex(const char* str) {
  while (*str) {
    // Print each character in HEX format
    //Serial.print("0x");
    if (*str < 0x10) {
      Serial.print("0"); // Add leading zero for single-digit HEX numbers
    }
    Serial.print(*str, HEX);
    Serial.print(" ");
    str++;
  }
  Serial.println(); // Move to the next line after printing the entire string
}

String requestAndReadData(const char* command) {
  
  size_t length = strlen(command);
  char cmd[length+2];
  sprintf(cmd,"%s\r\n",command);

  #if DEBUG
    Serial.print("Command '"); Serial.print(command); Serial.print("': ");
    printStringAsHex(cmd);
  #endif
  mySerial.write(cmd); // Send the SDI-12 command
  delay(50);                   // Wait for response to be ready
  
  String sensorResponse = "";
  //Read the response from the sensor
  while (mySerial.available()) { // Check if there is data available to read
    char c = mySerial.read();    // Read a single character
    if (c != -1) {              // Check if the character is valid
      sensorResponse += c;      // Append the character to the response string
      #if DEBUG
        Serial.print(c, HEX); Serial.print(" ");
      #endif
    }
    delay(20);  // otherwise it would leave some chars to next message...
  }
  // #if DEBUG
  //   Serial.println("");
  // #endif

  #if DEBUG
    char string_buffer[128]; // Buffer to hold the formatted string
    snprintf(string_buffer, sizeof(string_buffer), "response %s: %s\n", command, sensorResponse.c_str());
    Serial.println(string_buffer);
  #endif

  return sensorResponse;
}

// String requestAndReadData(String command, bool trim = false) {
//   mySDI12.sendCommand(command); // Send the SDI-12 command
//   delay(50);                   // Wait for response to be ready
  
//   String sensorResponse = "";
//   //Read the response from the sensor
//   while (mySDI12.available()) { // Check if there is data available to read
//     char c = mySDI12.read();    // Read a single character
//     if (c != -1) {              // Check if the character is valid
//       sensorResponse += c;      // Append the character to the response string
//       #if DEBUG
//         Serial.print(c, HEX); Serial.print(" ");
//       #endif
//     }
//     delay(20);  // otherwise it would leave some chars to next message...
//   }
//   #if DEBUG
//     Serial.println("");
//   #endif
// // >> 30 31 33 44 65 6C 74 61 2D 54 20 50 52 32 53 44 49 31 2E 31 50 52 32 2F 36 2D 30 34 35 30 36 30 D A command ?I!: 013Delta-T PR2SDI1.1PR2/6-045060
// // >> 30 30 30 32 36 D A command 0M!: 00026
// // >> 30 D A 30 2B 30 2E 39 38 32 2B 30 2E 39 38 2B 31 2E 30 30 34 38 2B 30 2E 39 38 35 34 2B 30 2E 39 38 37 35 D command 0D0!: 00+0.982+0.98+1.0048+0.9854+0.9875
// // >> A 30 2B 30 2E 39 38 30 32 D A command 0D1!: 0+0.9802
// // >> DATA: 00+0.982+0.98+1.0048+0.9854+0.98750+0.9802

//   // command 0D0! returns <address><CRLF><data><CRLF> => readStringUntil would need to called twice
//   // String sensorResponse = mySDI12.readStringUntil('\n');
  
//    // Replace \r and \n with empty strings
//   // sensorResponse.replace("\r", "");
//   // sensorResponse.replace("\n", "");
//   if (trim)
//     sensorResponse.trim();  // remove CRLF at the end

//   #if DEBUG
//     char string_buffer[128]; // Buffer to hold the formatted string
//     snprintf(string_buffer, sizeof(string_buffer), "command %s: %s", command.c_str(), sensorResponse.c_str());
//     Serial.println(string_buffer);
//   #endif

//   return sensorResponse;
// }

// String measureString(String measure_command, uint8_t address)
// {
//     measure_command = String(address) + measure_command + "!";
//     String measureResponse = requestAndReadData(measure_command);  // Command to take a measurement
//     // Serial.println(measureResponse);
//     delay(2000);

//     // for C commands
//     uint8_t position = 0;
//     String data_command = String(address) + "D" + String(position) + "!";
//     String sensorResponse = requestAndReadData(data_command);

//     // for M commands - multiple D commands
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
//     return sensorResponse;
// }

bool human_print = false;
void loop() {
  
  // Serial.println("TICK");

  // DDI on power on
  // if (mySerial.available()) { // Check if there is data available to read
  //   char c = mySerial.read();    // Read a single character
  //   Serial.println(c, HEX);
  // }
  // delay(20);

  String sensorResponse = "";

  /*********************************** TEROS31 tryouts **************************************/

  // sensorResponse = requestAndReadData("0A3!");  // Command to change address from 0 to 3
  // Serial.println(sensorResponse);
  
  // delay(1000);

  // sensorResponse = requestAndReadData("?!");  // Which address is connected?
  // sensorResponse = requestAndReadData("?I!");  // Command to get sensor info
  // sensorResponse = requestAndReadData("0I!");  // Command to get sensor at address 0 info
  // sensorResponse = requestAndReadData("0R0!");  // Command to get sensor info
  // Serial.println(sensorResponse);

  // SELECT ending characters <CR><LF>, i.e. "\r\n", inside the function 
  sensorResponse = requestAndReadData("?!");  // Which address is connected?
  delay(1000);
  sensorResponse = requestAndReadData("?I!");  // Command to get sensor info
  delay(1000);
  sensorResponse = requestAndReadData("0I!");  // Command to get sensor info at address 0
  delay(1000);
  sensorResponse = requestAndReadData("0R0!");  // Command for continuous reading (delay between commands at least 2s)
  delay(2500);

  
  sensorResponse = requestAndReadData("1I!");  // Command to get sensor at address 0 info
  delay(1000);
  sensorResponse = requestAndReadData("1R0!");  // Command to get sensor info
  delay(2500);
  
  sensorResponse = requestAndReadData("3I!");  // Command to get sensor at address 0 info
  delay(1000);
  sensorResponse = requestAndReadData("3R0!");  // Command to get sensor info
  delay(2500);

  // sensorResponse = ReadData();
  // delay(1000);

  /*********************************** TEROS31 tryouts **************************************/





  // float values[6];
  // size_t max_values = 10;
  // uint8_t num_values = 0;
  // measure("M",0,values,max_values,&num_values);
  
  // sensorResponse = measureString("C",0);
  // //sensorResponse = sensorResponse.substring(3); // first 3 characters are <address><CR><LF> => remove
  // sensorResponse = sensorResponse.substring(1); // first 1 characters is <address> => remove
  // if(human_print) Serial.print("permitivity:    ");
  // Serial.println(sensorResponse);

  // delay(1000);

  // sensorResponse = measureString("C1",0);
  // sensorResponse = sensorResponse.substring(1); // first 1 characters is <address> => remove
  // if(human_print) Serial.print("soil moisture:  ");
  // Serial.println(sensorResponse);

  // sensorResponse = measureString("C8",0);
  // sensorResponse = sensorResponse.substring(1); // first 1 characters is <address> => remove
  // if(human_print) Serial.print("millivolts:     ");
  // Serial.println(sensorResponse);

  // sensorResponse = measureString("C9",0);
  // sensorResponse = sensorResponse.substring(1); // first 1 characters is <address> => remove
  // if(human_print) Serial.print("raw ADC:        ");
  // Serial.println(sensorResponse);

  // char string_buffer[128]; // Buffer to hold the formatted string
  // snprintf(string_buffer, sizeof(string_buffer), "DATA: %s", sensorResponse.c_str());
  // Serial.println(string_buffer);
}