#ifndef FILE_INFO_H
#define FILE_INFO_H

// #include "FS.h"  // use directly SD instead of arbitrary FS (due to Arduino SD.h lib)
#include <SD.h>
#include "common.h"
using namespace hlavo;

#if defined(ESP32) || defined(ESP8266)
  // #define FILE_WRITE (O_RDONLY | O_WRONLY | O_CREAT)
  // #define FILE_APPEND (O_RDONLY | O_WRONLY | O_CREAT | O_APPEND)
#else // define the same for Arduino
  #define FILE_WRITE (O_READ | O_WRITE | O_CREAT)
  #define FILE_APPEND (O_READ | O_WRITE | O_CREAT | O_APPEND)
#endif

class FileInfo
{
    public:
        FileInfo(const char * path);
        virtual ~FileInfo(void)
            {}

        char* getPath();
        bool exists();
        void write(const char * message);
        void append(const char * message);
        void read();
        // void rename(const char * new_path);  // not available in Arduino SD.h
        void remove();
        FileInfo copy(const char * new_path);
    
    private:
        char _path[max_filepath_length];
};

FileInfo::FileInfo(const char * path)
{
    snprintf(_path, sizeof(_path),"%s", path);
}

char* FileInfo::getPath()
{
  return _path;
}

bool FileInfo::exists(){
  return SD.exists(_path);
}

void FileInfo::write(const char * message){
    // Serial.printf("Writing file: %s\n", _path);

    File file = SD.open(_path, FILE_WRITE);
    if(!file){
        Serial.println("Failed to open file for writing");
        return;
    }
    if(file.print(message)){
        // Serial.println("File written");
    } else {
        Serial.println("Write failed");
    }
    file.close();
}

void FileInfo::append(const char * message){
    //Serial.printf("Appending to file: %s\n", _path);

    File file = SD.open(_path, FILE_APPEND);
    if(!file){
        Serial.println("Failed to open file for appending");
        return;
    }
    if(file.print(message)){
        //Serial.println("Message appended");
    } else {
        Serial.println("Append failed");
    }
    file.close();
}

void FileInfo::read(){
    // Serial.printf("Reading file: %s\n", _path);

    File file = SD.open(_path);
    if(!file){
        Serial.println("Failed to open file for reading");
        return;
    }

    // Serial.print("Read from file: ");
    while(file.available()){
        Serial.write(file.read());
    }
    file.close();
}

// void FileInfo::rename(const char * new_path){
//     hlavo::SerialPrintf(300, "Renaming file %s to %s\n", _path, new_path);
//     if (SD.rename(_path, new_path)) {
//         snprintf(_path, sizeof(_path),"%s", new_path);
//         // Serial.println("File renamed");
//     } else {
//         Serial.println("Rename failed");
//     }
// }

void FileInfo::remove(){
    hlavo::SerialPrintf(300, "Deleting file: %s\n", _path);
    if(SD.remove(_path)){
        // Serial.println("File deleted");
    } else {
        Serial.println("Delete failed");
    }
}

FileInfo FileInfo::copy(const char * new_path)
{
  hlavo::SerialPrintf(300,"Copying file: %s to %s\n", _path, new_path);

  // Open source file
  File sourceFile = SD.open(_path, FILE_READ);
  if (!sourceFile) {
    Serial.println("Failed to open source file");
    return FileInfo("");
  }

  // Open destination file
  File destinationFile = SD.open(new_path, FILE_WRITE);
  if (!destinationFile) {
    Serial.println("Failed to open destination file");
    sourceFile.close(); // Close the source file before returning
    return FileInfo("");
  }

  // Copy contents from source file to destination file
  while (sourceFile.available()) {
    destinationFile.write(sourceFile.read());
  }

  // Close files
  sourceFile.close();
  destinationFile.close();
  // Serial.println("Copying finished.");
  return FileInfo(new_path);
}



#endif //FILE_INFO_H

