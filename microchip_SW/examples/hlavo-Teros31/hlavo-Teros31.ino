#include "sdi12_comm.h"
// #include "pr2_data.h"
#include "teros31_reader.h"

#include "Every.h"

#define SERIAL_BAUD 115200 /*!< The baud rate for the output serial port */
#define SDI12_DATA_PIN 4   /*!< The pin of the SDI-12 data bus */
#define POWER_PIN 47       /*!< The sensor power pin (or -1 if not switching power) */

/** Define the SDI-12 bus */
SDI12Comm sdi12_comm(SDI12_DATA_PIN, 3);

const uint8_t n_sdi12_sensors = 3;
const char sdi12_addresses[n_sdi12_sensors] = {'A','B','C'};  // sensor addresses on SDI-12, supported by Teros31: 0-9, A-Z

Teros31Reader teros31_readers[3] = {
  Teros31Reader(&sdi12_comm, sdi12_addresses[0]),
  Teros31Reader(&sdi12_comm, sdi12_addresses[1]),
  Teros31Reader(&sdi12_comm, sdi12_addresses[2])
};
uint8_t iss = 0;  // current sensor reading
bool teros31_all_finished = false;

Every timer_L1(20*1000);

// use Teros31 reader to request and read data from Teros31
// minimize delays so that it does not block main loop
void collect_and_write_Teros31()
{
  bool res = false;
  res = teros31_readers[iss].TryRequest();
  if(!res)  // failed request
  {
    teros31_readers[iss].Reset();
    iss++;

    if(iss >= n_sdi12_sensors)
      iss = 0;
    return;
  }

  teros31_readers[iss].TryRead();
  if(teros31_readers[iss].finished)
  {
    // DateTime dt = rtc_clock.now();
    // pr2_readers[iss].data.datetime = dt;
    // if(VERBOSE >= 1)
    {
      // Serial.printf("DateTime: %s. Writing PR2Data[a%d].\n", dt.timestamp().c_str(), pr2_addresses[iss]);
      char msg[400];
      hlavo::SerialPrintf(sizeof(msg)+20, "Teros31[%c]: %s\n",sdi12_addresses[iss], teros31_readers[iss].data.print(msg, sizeof(msg)));
    }

    // Logger::print("collect_and_write_PR2 - CSVHandler::appendData");
    // CSVHandler::appendData(data_spi_filenames[iss], &(pr2_readers[iss].data));

    teros31_readers[iss].Reset();
    iss++;
    if(iss == n_sdi12_sensors)
    {
      iss = 0;
      teros31_all_finished = true;
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

  delay(1000);  // allow things to settle
  uint8_t nbytes = 0;

  // CHANGE ADDRESS
  // String si = sdi12_comm.requestAndReadData("0Aa!", &nbytes);  // Command to get sensor info
  // String si = sdi12_comm.requestAndReadData("1A0!", false);  // Command to get sensor info
  // Logger::print(sdi12_comm.requestAndReadData("7AA!", &nbytes));  // Command to get sensor info
  // String si = sdi12_comm.requestAndReadData("0AB!", &nbytes);  // Command to get sensor info
  // String si = sdi12_comm.requestAndReadData("0AC!", &nbytes);  // Command to get sensor info
  delay(1000);

  // while(1)
  //   ;

  for(int i=0; i<n_sdi12_sensors; i++){
    String cmd = String(sdi12_addresses[i]) + "I!";
    Logger::print(sdi12_comm.requestAndReadData(cmd.c_str(), &nbytes));  // Command to get sensor info
  }

  Serial.println("Setup complete.");
  Serial.flush();
}

void loop() {

  delay(200);
  Serial.println("TICK");

  // uint8_t nbytes = 0;
  // String si = sdi12_comm.requestAndReadData("?I!", &nbytes);  // Command to get sensor info
  // Serial.println(si);

  // delay(300);

  uint8_t nbytes = 0;
  for(int i=0; i<n_sdi12_sensors; i++){
    String cmd = String(sdi12_addresses[i]) + "I!";
    Logger::print(sdi12_comm.requestAndReadData(cmd.c_str(), &nbytes));  // Command to get sensor info
    delay(1000);
  }

  // if(!teros31_all_finished){
  //   collect_and_write_Teros31();
  // }

  // if(timer_L1() && teros31_all_finished){
  //   teros31_all_finished = false;
  //   Serial.println("=============================================================================================");
  // }

}