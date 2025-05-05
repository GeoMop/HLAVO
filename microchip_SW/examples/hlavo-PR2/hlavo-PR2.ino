#include "sdi12_comm.h"
#include "pr2_data.h"
#include "pr2_reader.h"

#include "Every.h"

#define SERIAL_BAUD 115200 /*!< The baud rate for the output serial port */
#define SDI12_DATA_PIN 4   /*!< The pin of the SDI-12 data bus */
#define POWER_PIN 47       /*!< The sensor power pin (or -1 if not switching power) */
#define PR2_POWER_PIN 7        // The pin PR2 power

/** Define the SDI-12 bus */
SDI12Comm sdi12_comm(SDI12_DATA_PIN, 3);

const uint8_t n_sdi12_sensors = 3;
const char sdi12_addresses[n_sdi12_sensors] = {'0','1','3'};  // sensor addresses on SDI-12

PR2Reader pr2_readers[3] = {
  PR2Reader(&sdi12_comm, sdi12_addresses[0]),
  PR2Reader(&sdi12_comm, sdi12_addresses[1]),
  PR2Reader(&sdi12_comm, sdi12_addresses[2])
};
uint8_t iss = 0;  // current sensor reading
bool pr2_all_finished = false;

Every timer_L1(20*1000);

// use PR2 reader to request and read data from PR2
// minimize delays so that it does not block main loop
void collect_and_write_PR2()
{
  bool res = false;
  res = pr2_readers[iss].TryRequest();
  if(!res)  // failed request
  {
    pr2_readers[iss].Reset();
    iss++;

    if(iss >= n_sdi12_sensors)
      iss = 0;
    return;
  }

  pr2_readers[iss].TryRead();
  if(pr2_readers[iss].finished)
  {
    // DateTime dt = rtc_clock.now();
    // pr2_readers[iss].data.datetime = dt;
    // if(VERBOSE >= 1)
    {
      // Serial.printf("DateTime: %s. Writing PR2Data[a%d].\n", dt.timestamp().c_str(), pr2_addresses[iss]);
      char msg[400];
      hlavo::SerialPrintf(sizeof(msg)+20, "PR2[%c]: %s\n",sdi12_addresses[iss], pr2_readers[iss].data.print(msg, sizeof(msg)));
    }

    // Logger::print("collect_and_write_PR2 - CSVHandler::appendData");
    // CSVHandler::appendData(data_spi_filenames[iss], &(pr2_readers[iss].data));

    pr2_readers[iss].Reset();
    iss++;
    if(iss == n_sdi12_sensors)
    {
      iss = 0;
      pr2_all_finished = true;
    }
  }
}


void setup() {
  Serial.begin(SERIAL_BAUD);
  while (!Serial)
    ;

  Serial.println("Opening SDI-12 bus...");
  sdi12_comm.begin();
  delay(500);  // allow things to settle

  // Power the sensors;
  if (POWER_PIN > 0) {
    Serial.println("Powering up sensors...");
    pinMode(POWER_PIN, OUTPUT);
    digitalWrite(POWER_PIN, HIGH);
    delay(500);
  }

  // PR2
  pinMode(PR2_POWER_PIN, OUTPUT);
  setPin(PR2_POWER_PIN, HIGH);  // turn on power for PR2

  // CHANGE ADDRESS
  // String si = sdi12_comm.requestAndReadData("0A1!", false);  // Command to get sensor info
  // String si = sdi12_comm.requestAndReadData("1A0!", false);  // Command to get sensor info

  delay(1000);  // allow things to settle
  uint8_t nbytes = 0;
  for(int i=0; i<n_sdi12_sensors; i++){
    String cmd = String(sdi12_addresses[i]) + "I!";
    Logger::print(sdi12_comm.requestAndReadData(cmd.c_str(), &nbytes));  // Command to get sensor info
  }

  Serial.flush();
}

bool human_print = true;


// Sequential "blocking" read
void read_pr2(uint8_t address)
{
  float values[10];
  uint8_t n_values = 0;
  String sensorResponse = "";

  delay(300);
  String info_cmd = String(address) + "I!";
  uint8_t nbytes = 0;
  Serial.println(info_cmd);
  String si = sdi12_comm.requestAndReadData(info_cmd.c_str(), &nbytes);  // Command to get sensor info
  delay(300);

  bool res = false;
  sensorResponse = sdi12_comm.measureRequest("C", address, &res);
  Serial.println(sensorResponse);
  // delay(sdi12_delay_timer.interval);
  // sensorResponse = sdi12_comm.measureRead(address, values, &n_values);
  // Serial.println(sensorResponse);
  // sdi12_comm.print_values("permitivity", values, n_values);

  // sensorResponse = sdi12_comm.measureRequestAndRead("C", address, values, &n_values);
  // sdi12_comm.print_values("permitivity", values, n_values);

  // sensorResponse = pr2.measureRequestAndRead("C1", address, values, &n_values);
  // pr2.print_values("soil moisture mineral", values, n_values);

  // sensorResponse = pr2.measureRequestAndRead("C2", address, values, &n_values);
  // pr2.print_values("soil moisture organic", values, n_values);

  // sensorResponse = pr2.measureRequestAndRead("C3", address, values, &n_values);
  // pr2.print_values("soil moisture mineral (%)", values, n_values);

  // sensorResponse = pr2.measureRequestAndRead("C4", address, values, &n_values);
  // pr2.print_values("soil moisture mineral (%)", values, n_values);

  // sensorResponse = pr2.measureRequestAndRead("C7", address, values, &n_values);
  // pr2.print_values("millivolts", values, n_values);

  // sensorResponse = pr2.measureRequestAndRead("C8", address, values, &n_values);
  // pr2.print_values("millivolts uncalibrated", values, n_values);

  // sensorResponse = pr2.measureRequestAndRead("C9", address, values, &n_values);
  // pr2.print_values("raw ADC", values, n_values);
}




void loop() {

  delay(200);
  Serial.println("TICK");

  // String sensorResponse = "";

  delay(300);

  // String si = pr2.requestAndReadData("?I!", false);  // Command to get sensor info

  // uint8_t nbytes = 0;
  // String si = sdi12_comm.requestAndReadData("0I!", &nbytes);  // Command to get sensor info
  // Serial.println(si);

  // delay(1000);

  // si = sdi12_comm.requestAndReadData("1I!", &nbytes);  // Command to get sensor info
  // Serial.println(si);

  delay(1000);

  // Serial.println("---------------------------------------------------- address 0");
  // read_pr2(0);
  // Serial.println("---------------------------------------------------- address 1");
  read_pr2(1);
  // Serial.println("---------------------------------------------------- address 3");
  // read_pr2(3);

  // if(!pr2_all_finished){
  //   collect_and_write_PR2();
  // }

  // if(timer_L1() && pr2_all_finished)
  //   pr2_all_finished = false;

  // {
  //   pr2_readers[iadd].TryRequest();
  //   pr2_readers[iadd].TryRead();
  //   if(pr2_readers[iadd].finished)
  //   {
  //     char csvLine[200];
  //     Serial.println(pr2_readers[iadd].data.dataToCsvLine(csvLine, 200));
  //     pr2_readers[iadd].Reset();

  //     iadd++;
  //     if(iadd == 2)
  //     {
  //       iadd = 0;
  //     }
  //   }
  // }

  {
    // pr2_reader.TryRequest();
    // pr2_reader.TryRead();
    // if(pr2_reader.finished)
    // {
    //   char csvLine[200];
    //   Serial.println(pr2_reader.data.dataToCsvLine(csvLine));
    //   pr2_reader.Reset();
    // }
  }

  {
    // uint8_t address = 0;
    // float values[10];
    // uint8_t n_values = 0;
    // String sensorResponse = "";

    // if(!pr2_delay_timer.running)
    //   sensorResponse = pr2.measureRequest("C", address);
    // if(pr2_delay_timer())
    // {
    //   sensorResponse = pr2.measureRead(address, values, &n_values);
    //   pr2.print_values("permitivity", values, n_values);
    // }
  }

  // Serial.println("---------------------------------------------------- address 1");
  


  // sensorResponse = requestAndReadData("?I!", true);  // Command to get sensor info
  // //Serial.println(sensorResponse);

  // sensorResponse = measureString("C",0);
  // //sensorResponse = sensorResponse.substring(3); // first 3 characters are <address><CR><LF> => remove
  // // sensorResponse = sensorResponse.substring(1); // first 1 characters is <address> => remove
  // Serial.print("permitivity:    "); Serial.println(sensorResponse);

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




  // //Info
  // Serial.printf("Command %s:", myCommand);
  // mySDI12.sendCommand(myCommand);
  // delay(300);                    // wait a while for a response
  // while (mySDI12.available()) {  // write the response to the screen
  //   Serial.write(mySDI12.read());
  // }

  // // const char* cmd1 = "?!";
  // // const char* cmd2 = "0M!";
  // // const char* cmd3 = "0D0!";

  // // Address
  // Serial.printf("Command %s:", myCommand1);
  // mySDI12.sendCommand(myCommand1);
  // // mySDI12.sendCommand(cmd1);
  // delay(300);                    // wait a while for a response
  // // Serial.print("Address: ");
  // String sensorResponse="";
  // while (mySDI12.available()) {  // write the response to the screen
  //   char c = mySDI12.read();
  //   Serial.print(c, HEX); Serial.print(" ");
  //   if (c != -1) {              // Check if the character is valid
  //     sensorResponse += c;      // Append the character to the response string
  //     #if DEBUG
  //       Serial.print(c, HEX); Serial.print(" ");
  //     #endif
  //   }
  //   delay(20);  // otherwise it would leave some chars to next message...
  // //  Serial.write(mySDI12.read());
  //   // char buf[25];
  //   // Serial.printf("%s-", itoa(mySDI12.read(), buf, 10));
  // }
  // Serial.print(sensorResponse);
  // Serial.println();
  // delay(500); 

  // // M7
  // // Serial.printf("Command %s:", "0M7!");
  // // mySDI12.sendCommand("0M7!");
  // // // mySDI12.sendCommand(cmd1);
  // // delay(300);                    // wait a while for a response
  // // // Serial.print("Address: ");
  // // while (mySDI12.available()) {  // write the response to the screen
  // //  Serial.write(mySDI12.read());
  // //   // char buf[25];
  // //   // Serial.printf("%s-", itoa(mySDI12.read(), buf, 10));
  // // }
  // // Serial.println();
  // // delay(500); 


  // // Measure
  //   String cmd = "0C!";
  //   // String cmd = "0C!";
  //   Serial.printf("Command %s:", cmd);
  //   mySDI12.sendCommand(cmd);
  //   delay(300);                    // wait a while for a response
  //   while (mySDI12.available()) {  // build response string
  //       char c = mySDI12.read();
  //       Serial.print(c, HEX); Serial.print(" ");
  //       if ((c != '\n') && (c != '\r'))
  //       {
  //         sdiResponse += c;
  //       }
  //       delay(20);  // 1 character ~ 7.5ms
  //   }
  //   if (sdiResponse.length() > 1)
  //       Serial.println(sdiResponse);  // write the response to the screen
  //   mySDI12.clearBuffer();
  //   sdiResponse = "";  // clear the response string

  //   delay(2000); 

  //   // DATA
  //   Serial.print("Command 0D0!: ");
  //   mySDI12.sendCommand(myCommand3);
  //   // mySDI12.sendCommand(cmd3);
  //   delay(3000);                    // wait a while for a response
    
  //   while (mySDI12.available()) {  // build response string
  //       char c = mySDI12.read();
  //       Serial.print(c, HEX); Serial.print(" ");
  //   // if (c != -1) {              // Check if the character is valid
  //   //   sensorResponse += c;      // Append the character to the response string
  //   //   #if DEBUG
  //   //     Serial.print(c, HEX); Serial.print(" ");
  //   //   #endif
  //   // }
  //   // delay(20);  // otherwise it would leave some chars to next message...

  //       if ((c != '\n') && (c != '\r'))
  //       {
  //         sdiResponse += c;
  //         // char buf[25];
  //         // Serial.printf("%s-", itoa(mySDI12.read(), buf, 10));
  //       }
  //       delay(20);  // 1 character ~ 7.5ms
  //   }
  //   if (sdiResponse.length() > 1)
  //       Serial.println(sdiResponse);  // write the response to the screen
  //   // Serial.println();
  //   mySDI12.clearBuffer();
  //   sdiResponse = "";  // clear the response string

















//   mySDI12.sendCommand(myCommand2);
//   delay(300);                    // wait a while for a response
//   Serial.print("Command 0M!: ");
//   while (mySDI12.available()) {  // write the response to the screen
//   Serial.write(mySDI12.read());
//   }
  
//   delay(1000); 

//   mySDI12.sendCommand(myCommand3);
//   delay(300);                    // wait a while for a response
//   Serial.print("Command 0D0!: ");
//   while (mySDI12.available()) {  // write the response to the screen
//   Serial.write(mySDI12.read());
//   }
//   delay(2000); 

//   mySDI12.sendCommand(myCommand4);
//   delay(300);                    // wait a while for a response
//   Serial.print("Command 0D1!: ");
//   while (mySDI12.available()) {  // write the response to the screen
//   Serial.write(mySDI12.read());
//   }
//    delay(500); 







//   mySDI12.sendCommand(myCommand5);
//   delay(300);                    // wait a while for a response
//   Serial.print("Command 0M1!: ");
//   while (mySDI12.available()) {  // write the response to the screen
//   Serial.write(mySDI12.read());
//   }
//   delay(2000); 

//   mySDI12.sendCommand(myCommand6);
//   delay(300);                    // wait a while for a response
//   Serial.print("Command 0D0!: ");
//   while (mySDI12.available()) {  // write the response to the screen
//   Serial.write(mySDI12.read());
//   }
//   delay(500); 

//   mySDI12.sendCommand(myCommand7);
//   delay(300);                    // wait a while for a response
//   Serial.print("Command 0D1!: ");
//   while (mySDI12.available()) {  // write the response to the screen
//   Serial.write(mySDI12.read());
//   }
//   delay(500); 

//   mySDI12.sendCommand(myCommand8);
//   delay(300);                    // wait a while for a response
//   Serial.print("Command 0M8!: ");
//   while (mySDI12.available()) {  // write the response to the screen
//   Serial.write(mySDI12.read());
//   }
//   delay(2000); 

//   mySDI12.sendCommand(myCommand9);
//   delay(300);                    // wait a while for a response
//   Serial.print("Command 0D0!: ");
//   while (mySDI12.available()) {  // write the response to the screen
//   Serial.write(mySDI12.read());
//   }
//   delay(500); 

//   mySDI12.sendCommand(myCommand10);
//   delay(300);                    // wait a while for a response
//   Serial.print("Command 0D1!: ");
//   while (mySDI12.available()) {  // write the response to the screen
//   Serial.write(mySDI12.read());
//   }
//   delay(500);
// mySDI12.sendCommand(myCommand11);
//   delay(300);                    // wait a while for a response
//   Serial.print("Command Raw ADC: ");
//   while (mySDI12.available()) {  // write the response to the screen
//   Serial.write(mySDI12.read());
//   }
//   delay(2000); 

//   mySDI12.sendCommand(myCommand12);
//   delay(300);                    // wait a while for a response
//   Serial.print("Command 0D0!: ");
//   while (mySDI12.available()) {  // write the response to the screen
//   Serial.write(mySDI12.read());
//   }
//   delay(500); 

//   mySDI12.sendCommand(myCommand13);
//   delay(300);                    // wait a while for a response
//   Serial.print("Command 0D1!: ");
//   while (mySDI12.available()) {  // write the response to the screen
//   Serial.write(mySDI12.read());
//   }
//   delay(500);
//   mySDI12.sendCommand(myCommand14);
//   delay(300);                    // wait a while for a response
//   Serial.print("Command Permitivity: ");
//   while (mySDI12.available()) {  // write the response to the screen
//   Serial.write(mySDI12.read());
//   }
//   delay(500); 

//   mySDI12.sendCommand(myCommand15);
//   delay(300);                    // wait a while for a response
//   Serial.print("Command Theta: ");
//   while (mySDI12.available()) {  // write the response to the screen
//   Serial.write(mySDI12.read());
//   }
//   delay(500);

//   mySDI12.sendCommand(myCommand16);
//   delay(300);                    // wait a while for a response
//   Serial.print("Command Perc Volumetric: ");
//   while (mySDI12.available()) {  // write the response to the screen
//   Serial.write(mySDI12.read());
//   }
  // delay(2000);  // print again in three seconds
}