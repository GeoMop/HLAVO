#ifndef HLAVO_LOGGER_H_
#define HLAVO_LOGGER_H_

#include <FreeRTOS.h>
#include <queue.h>

#include <RTClib.h>
#include <SD.h>
#include "file_info.h"
#include "clock.h"
#include <stdarg.h>
#include "common.h"

/// @brief Wrapper class for SD card and CSV file handling.
class Logger
{
  public:
    enum MessageType { INFO, WARN, ERROR };
    static void setup_log(Clock &clock, const char* dir_name);

    static void print(const char* msg, MessageType type = INFO);
    static void print(const String& msg, MessageType type = INFO);
    static void printHex(const char* data, size_t length, MessageType type = INFO);
    static void printf(MessageType type, const char* format, ...);

    static void cleanup_old_logs(int retentionDays = 7);

  private:
    static bool initialized;
    // static const int _retentionDays = 7;
    static Clock* _rtc_clock;
    static char _logDirectory[hlavo::max_dirpath_length];
    static FileInfo _logfile;
    // static File _logfile;
    static const int log_msg_maxsize = 500;
    static char _log_buf[log_msg_maxsize];

    static void sendToWriteTask();
    static void writeTask(void *pvParameters);
    static void clean_buf();
    static void createLogFileName();
    static String messageTypeToString(MessageType type);

    static QueueHandle_t messageQueue;
};

// Static member initialization
Clock* Logger::_rtc_clock = nullptr;
char Logger::_logDirectory[hlavo::max_dirpath_length] = "/logs";
FileInfo Logger::_logfile = FileInfo("");
// File Logger::_logfile = File();
char Logger::_log_buf[log_msg_maxsize] = "";
bool Logger::initialized = false;
QueueHandle_t Logger::messageQueue = nullptr;



void Logger::setup_log(Clock &clock, const char* dir_name)
{
  _rtc_clock = &clock;

  snprintf(_logDirectory, hlavo::max_dirpath_length, "/%s", dir_name);

  if (!SD.exists(_logDirectory)) {
      SD.mkdir(_logDirectory);
  }

  createLogFileName();

  messageQueue = xQueueCreate(10, log_msg_maxsize * sizeof(char));
  if (messageQueue != NULL) {
      Serial.println("Logger messageQueue created.");
      // xTaskCreate(writeTask, "MessageWriter", 20480, NULL, 1, NULL);
      xTaskCreatePinnedToCore(writeTask, "MessageWriter", 4096, NULL, 0, NULL, 1);
  }
  delay(100);

  initialized = true;
}

void Logger::clean_buf()
{
  for(int i=0; i<log_msg_maxsize; i++)
    _log_buf[i] = '\0';
}

void Logger::sendToWriteTask() {
  // Serial.print(_log_buf);
  // _logfile.append(_log_buf);

  if (xQueueSend(messageQueue, (void *)_log_buf, portMAX_DELAY) == pdPASS) {
    // Serial.println("Log message sent!");
  }
}

void Logger::writeTask(void *pvParameters) {
  char receivedMessage[log_msg_maxsize];
  while (1) {
    // Receive data from the queue
    if (xQueueReceive(messageQueue, &receivedMessage, portMAX_DELAY) == pdTRUE) {

      // Serial.println("Log message received!");

      String dt_now = _rtc_clock->now().timestamp() + ": ";
      size_t newStringLength = dt_now.length();
      size_t originalLength = strlen(receivedMessage);

      memmove(receivedMessage + newStringLength, receivedMessage, originalLength + 1);
      memcpy(receivedMessage, dt_now.c_str(), newStringLength);

      // Append message to a file
      Serial.print(receivedMessage);
      _logfile.append(receivedMessage);

      // File file = SD.open(_path, FILE_APPEND);
      // if(!_logfile){
      //     Serial.println("ERROR: Log file not opened!");
      //     return;
      // }
      // if(_logfile.print(receivedMessage)){
      //     //Serial.println("Message appended");
      // } else {
      //     Serial.println("ERROR while appending log file.");
      // }
      // _logfile.close();
    }
    // delay(200);
  }
}

void Logger::print(const char* msg, MessageType type) {

    clean_buf();

    if(!initialized)
    {
      // keep serial output
      snprintf(_log_buf, sizeof(_log_buf), "[%s] %s\n", messageTypeToString(type).c_str(), msg);
      Serial.print(_log_buf);
      return;
    }

    // DateTime now = DateTime(2024, 10, 9, 6, 7, 8);
    // DateTime now = _rtc_clock->now();
    // snprintf(_log_buf, sizeof(_log_buf), "%s: [%s] %s\n", now.timestamp().c_str(), messageTypeToString(type).c_str(), msg);

    snprintf(_log_buf, sizeof(_log_buf), "[%s] %s\n", messageTypeToString(type).c_str(), msg);
    sendToWriteTask();
}

void Logger::print(const String& msg, MessageType type) {
    print(msg.c_str(), type);
}

void Logger::printHex(const char* data, size_t length, MessageType type) {
    // String hexString = "HEX: ";
    // char buf[4]; // Enough to store "FF "
    // for (size_t i = 0; i < length; i++) {
    //     snprintf(buf, sizeof(buf), "%02X ", static_cast<unsigned char>(data[i]));
    //     hexString += buf;
    // }

    if (data == nullptr || length == 0) return; // Validate input

    // Calculate buffer size (3 chars per byte for hex and space, plus "HEX: " prefix and a null terminator)
    size_t bufferSize = 5 + length * 3 + 1;
    char hexString[bufferSize]; // Allocate buffer on stack

    strcpy(hexString, "HEX: "); // Start with the prefix

    char* bufPtr = hexString + 5; // Pointer to place hex values after prefix
    for (size_t i = 0; i < length; i++) {
        snprintf(bufPtr, 4, "%02X ", (unsigned char)data[i]); // Print each byte in hex format
        bufPtr += 3; // Move the pointer forward by 3 positions ("FF ")
    }
    if (length > 0) {
        *(bufPtr - 1) = '\0'; // Replace the last space with a null terminator
    } else {
        *bufPtr = '\0'; // Ensure string is null-terminated
    }

    print(hexString, type);
}

void Logger::printf(MessageType type, const char* format, ...) {

    clean_buf();
    va_list args;
    va_start(args, format);

    if(!initialized)
    {
      char head[50];
      snprintf(head, sizeof(head), "[%s] ", messageTypeToString(type).c_str());
      snprintf(_log_buf, sizeof(_log_buf), "%s", head);
      size_t offset = strlen(_log_buf);
      vsnprintf(_log_buf + offset, sizeof(_log_buf)-offset, format, args);

      va_end(args);
      Serial.println(_log_buf);
      return;
    }

    // // print start
    // DateTime now = DateTime(2024, 10, 9, 6, 7, 8);
    // DateTime now = _rtc_clock->now();
    char head[50];
    // snprintf(head, sizeof(head), "%s: [%s] ", now.timestamp().c_str(), messageTypeToString(type).c_str());
    snprintf(head, sizeof(head), "[%s] ", messageTypeToString(type).c_str());
    snprintf(_log_buf, sizeof(_log_buf), "%s", head);
    size_t offset = strlen(_log_buf);

    vsnprintf(_log_buf + offset, sizeof(_log_buf)-offset, format, args);

    va_end(args);

    sendToWriteTask();
}

void Logger::cleanup_old_logs(int retentionDays) {
    File root = SD.open(_logDirectory);
    if (!root) {
        Serial.println("Failed to open log directory.");
        return;
    }

    root.rewindDirectory();
    File file = root.openNextFile();
    while (file) {
        String fileName = String(file.name());
        if (fileName.endsWith("_hlavo_station.log")) {
            // Extract the timestamp from the file name
            String timestamp = fileName.substring(fileName.lastIndexOf('/') + 1, fileName.lastIndexOf('_'));
            // Parse the date (format: YYYYMMDD)
            int year = timestamp.substring(0, 4).toInt();
            int month = timestamp.substring(4, 6).toInt();
            int day = timestamp.substring(6, 8).toInt();

            DateTime fileDate(year, month, day);
            DateTime currentDate = _rtc_clock->now();
            TimeSpan fileAge = currentDate - fileDate;

            if (fileAge.days() > retentionDays) {
                SD.remove(String(_logDirectory) + "/" + fileName);
                Serial.println("Deleted old log file: " + fileName);
            }
        }
        file = root.openNextFile();
    }
    root.close();
}



void Logger::createLogFileName() {
    DateTime now = _rtc_clock->now();
    char buf[hlavo::max_filepath_length];
    snprintf(buf, sizeof(buf), "%s/%04d%02d%02d_hlavo.log", _logDirectory, now.year(), now.month(), now.day());
    Serial.println(buf);
    _logfile = FileInfo(buf);
    // _logfile = SD.open(buf, FILE_APPEND);
    // if(!_logfile){
    //     Serial.println("Failed to open file for appending");
    //     return;
    // }
}

String Logger::messageTypeToString(MessageType type) {
    switch (type) {
        case INFO: return "INFO";
        case WARN: return "WARN";
        case ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

#endif // HLAVO_LOGGER_H_
