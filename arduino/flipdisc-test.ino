//SPI Initlization
#include <SPI.h>
#include <Arduino.h>

//Pin Defination
//Data pin
//Reset pin
//Row_latch pin
//Col_latch pin

//Use for flip dot
//Set_reset pin
//Pulse pin

//Define Variable
#define SPI_SPEED 8000000
#define SET 1
#define RESET 0
#define PULSE_LENGTH_US 200

void setup()
{

    // Initialize serial communication
    Serial.begin(115200);
    Serial.println("Flip-disc Test Initialized");
}

void loop()
{

}
