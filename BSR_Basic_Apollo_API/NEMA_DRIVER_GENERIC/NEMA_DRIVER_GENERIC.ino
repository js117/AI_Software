#include <Wire.h>
#include <Stepper.h>
// FIXING THE STEPPER LIB http://forum.arduino.cc/index.php?topic=46964.0
// C:\Program Files (x86)\Arduino\libraries\Stepper
/*

void Stepper::setSpeed(long whatSpeed)
{
 this->step_delay = 60L * 1000L / this->number_of_steps * 1000 / whatSpeed;     // THIS LINE
}

void Stepper::step(int steps_to_move)
{  
  if (micros() - this->last_step_time >= this->step_delay)                      // THIS LINE
     this->last_step_time = micros();                                           // THIS LINE
}


*/ 



// GENERIC NEMA STEPPER MOTOR DRIVER CONTROL VIA SERIAL

/*
Equipment:

1 Stepper Motor (e.g. @ 200 steps/rev)
1 ARDUINO UNO
1 power supply
1 Driver (M542T, MA860H, TB6560)
4 M-M Jumper Wires

Wire connection:

(Power supply: Green-GND; White-HOT; Black-NEUT -- double-check your cable before plugging into wall)

Pin 11: PUL-
Pin 10: DIR-
Pin  9: PUL+
Pin  8: DIR+

Program Info:

Accept the serial input @ 19200 buad 
*/
// initialize some constants and variables
int STEPS = 10; // control how many steps we want to turn for each call
int distance = 0; // number of steps elapsed
#define STEP_CHANGE 1 // number to change STEPS number
#define SPEED_CHANGE 20; // number to change SPEED
int motorSpeed = 1000; // initialize the motor speed: 60 rpm
int stepsPerRevolution = 200; // doesn't seem to matter much; 400 worked
// initialize the stepper library on pins 8, 10, 9, 11:
Stepper myStepper(stepsPerRevolution, 2, 4, 12, 13); 

// NOTE: stepper library works with any digital PWM pins. Just make sure the order is: 
// (.., DIR+, DIR-, PUL+, PUL-) with the motor driver.


// setup the devise
void setup() {
  // initialize the serial port:
  Serial.begin(19200);
  myStepper.setSpeed(motorSpeed);
  distance = 0;
}


// loop functoin
void loop() {
  
  if (Serial.available() > 0) {
        int recv = Serial.read();
        
        if (recv == 'q') {
          
            myStepper.step(STEPS);
            distance = distance + STEPS;
            
        } else if (recv == 'w') {
          
              myStepper.step(-STEPS);
              distance = distance - STEPS;
              
        } else if (recv == 'e') {                    // mostly for testing manually thru Arduino
          
              STEPS = STEPS + STEP_CHANGE;
              
        } else if (recv == 'r') {                    // mostly for testing manually thru Arduino
          
              STEPS = STEPS - STEP_CHANGE;
              STEPS = max(STEPS, 1);
            
        } else if (recv == 't') {
            
              motorSpeed = motorSpeed + SPEED_CHANGE;
              myStepper.setSpeed(motorSpeed);
        
        } else if (recv == 'y') {
              
              motorSpeed = motorSpeed - SPEED_CHANGE;
              motorSpeed = max(motorSpeed, 1);
              myStepper.setSpeed(motorSpeed);      
          
        } else if (recv == 'u') {                    // reset distance
        
              distance = 0;
          
        } else if (recv == 'i') {                    // i for info
          
          Serial.println("");
          Serial.println("----------------------");
          Serial.println("--- STATUS UPDATE: ---");
          Serial.println("----------------------");
          Serial.print("--- Motor Speed:   "); Serial.println(motorSpeed);
          Serial.print("--- Steps counter: "); Serial.println(distance);
          Serial.print("--- Steps per cmd: "); Serial.println(STEPS);
          Serial.println("----------------------");
          Serial.println("");
          
       }
  }
}


