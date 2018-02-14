
/* 
    Big Solve Robotics 
    
    Embedded Software for Arduino Mega Microcontroller Board
*/    
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
// NOTE: TB6560 can work like any other stepper driver
//       CLK+，CLK- Pulse positive and negative
//       CW+，CW- Direction positive and negative

#include <Wire.h>
#include <Stepper.h>

String inData;
String function;
unsigned long curMillis;

// Driver 1: Standard Microstepper
const int D1_Pp = 44; const int J1stepPin = D1_Pp;
const int D1_Pm = 46; 
const int D1_Dp = 48; const int J1dirPin = D1_Dp; 
const int D1_Dm = 50;
int D1_Dir = 1;
// Driver 2: Standard Microstepper 
//STUPID ERROR: using pins 0 and 1... these are for TTL serial data to the FTDI USB-to-TTL chip
const int D2_Pp = 12; const int J2stepPin = D2_Pp;
const int D2_Pm = 13;
const int D2_Dp = 2; const int J2dirPin = D2_Dp; 
const int D2_Dm = 3;
int D2_Dir = 1;
// Driver 3: Standard Microstepper
const int D3_Pp = 4; const int J3stepPin = D3_Pp;
const int D3_Pm = 5;
const int D3_Dp = 6; const int J3dirPin = D3_Dp; 
const int D3_Dm = 7;
int D3_Dir = 1;
// Driver 4: Standard Microstepper
const int D4_Pp = 8; const int J4stepPin = D4_Pp;
const int D4_Pm = 9;
const int D4_Dp = 10; const int J4dirPin = D4_Dp; 
const int D4_Dm = 11;
int D4_Dir = 1;
// Driver 5: Standard Microstepper
const int D5_Pp = 14; const int J5stepPin = D5_Pp;
const int D5_Pm = 15;
const int D5_Dp = 16; const int J5dirPin = D5_Dp; 
const int D5_Dm = 17;
int D5_Dir = 1;
// Driver 6: Standard Microstepper
const int D6_Pp = 18; const int J6stepPin = D6_Pp;
const int D6_Pm = 19;
const int D6_Dp = 20; const int J6dirPin = D6_Dp; 
const int D6_Dm = 21;
int D6_Dir = 1;
// Driver 7: Low-end/basic Microstepper
const int D7_Pp = 22; const int J7stepPin = D7_Pp;
const int D7_Pm = 24;
const int D7_Dp = 26; const int J7dirPin = D7_Dp; 
const int D7_Dm = 28;
int D7_Dir = 1;
// Driver 8: Low-end/basic Microstepper
const int D8_Pp = 30; const int J8stepPin = D8_Pp;
const int D8_Pm = 32;
const int D8_Dp = 34; const int J8dirPin = D8_Dp; 
const int D8_Dm = 36;
int D8_Dir = 1;



#define STEP_CHANGE 1 // number to change STEPS number
#define SPEED_CHANGE 10 // incremental number to change SPEED
#define SPEED_MAX 2510 // max RPMs for stepper
const int PulseWidth = 10;
float motorSpeed1 = 400.0; // initialize the motor speed
float motorSpeed2 = 400.0; // initialize the motor speed
float motorSpeed3 = 400.0; // initialize the motor speed
float motorSpeed4 = 400.0; // initialize the motor speed
float motorSpeed5 = 400.0; // initialize the motor speed
float motorSpeed6 = 400.0; // initialize the motor speed
float motorSpeed7 = 400.0; // need to normalize to milliesBetweenSteps variables below
float motorSpeed8 = 400.0; // need to normalize to milliesBetweenSteps variables below
float stepsPerRevolution1 = 200.0;
float stepsPerRevolution2 = 200.0;
float stepsPerRevolution3 = 200.0;
float stepsPerRevolution4 = 200.0;
float stepsPerRevolution5 = 200.0;
float stepsPerRevolution6 = 200.0;
float stepsPerRevolution7 = 200.0;
float stepsPerRevolution8 = 200.0;

// TODO: delays in timing for the different motors. 
// Basically instead of using num_steps 
// Default value: delay value for 1 whole revolution (200 steps). 
// Need to update dynamically when receiving new commands. Update both the steps, and speed.
// Meant for delayMicros()
float stepDelay1 = 60.0 * 1000.0 / stepsPerRevolution1 * 1000.0 / motorSpeed1;
float stepDelay2 = 60.0 * 1000.0 / stepsPerRevolution2 * 1000.0 / motorSpeed2;
float stepDelay3 = 60.0 * 1000.0 / stepsPerRevolution3 * 1000.0 / motorSpeed3;
float stepDelay4 = 60.0 * 1000.0 / stepsPerRevolution4 * 1000.0 / motorSpeed4;
float stepDelay5 = 60.0 * 1000.0 / stepsPerRevolution5 * 1000.0 / motorSpeed5;
float stepDelay6 = 60.0 * 1000.0 / stepsPerRevolution6 * 1000.0 / motorSpeed6;
float stepDelay7 = 60.0 * 1000.0 / stepsPerRevolution7 * 1000.0 / motorSpeed7;
float stepDelay8 = 60.0 * 1000.0 / stepsPerRevolution8 * 1000.0 / motorSpeed8;

// NOTE: stepper library works with any digital PWM pins. Just make sure the order is: 
// (.., DIR+, DIR-, PUL+, PUL-) with the motor driver.
Stepper D1_Stepper(stepsPerRevolution1, D1_Dp, D1_Dm, D1_Pp, D1_Pm); 
Stepper D2_Stepper(stepsPerRevolution2, D2_Dp, D2_Dm, D2_Pp, D2_Pm); 
Stepper D3_Stepper(stepsPerRevolution3, D3_Dp, D3_Dm, D3_Pp, D3_Pm); 
Stepper D4_Stepper(stepsPerRevolution4, D4_Dp, D4_Dm, D4_Pp, D4_Pm); 
Stepper D5_Stepper(stepsPerRevolution5, D5_Dp, D5_Dm, D5_Pp, D5_Pm); 
Stepper D6_Stepper(stepsPerRevolution6, D6_Dp, D6_Dm, D6_Pp, D6_Pm); 
Stepper D7_Stepper(stepsPerRevolution7, D7_Dp, D7_Dm, D7_Pp, D7_Pm); 
Stepper D8_Stepper(stepsPerRevolution8, D8_Dp, D8_Dm, D8_Pp, D8_Pm); 

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200); // 115200

  pinMode(D1_Pp, OUTPUT);
  pinMode(D1_Pm, OUTPUT);
  pinMode(D1_Dp, OUTPUT);
  pinMode(D1_Dm, OUTPUT);
  //
  pinMode(D2_Pp, OUTPUT);
  pinMode(D2_Pm, OUTPUT);
  pinMode(D2_Dp, OUTPUT);
  pinMode(D2_Dm, OUTPUT);
  //
  pinMode(D3_Pp, OUTPUT);
  pinMode(D3_Pm, OUTPUT);
  pinMode(D3_Dp, OUTPUT);
  pinMode(D3_Dm, OUTPUT);
  //
  pinMode(D4_Pp, OUTPUT);
  pinMode(D4_Pm, OUTPUT);
  pinMode(D4_Dp, OUTPUT);
  pinMode(D4_Dm, OUTPUT);
  //
  pinMode(D5_Pp, OUTPUT);
  pinMode(D5_Pm, OUTPUT);
  pinMode(D5_Dp, OUTPUT);
  pinMode(D5_Dm, OUTPUT);
  //
  pinMode(D6_Pp, OUTPUT);
  pinMode(D6_Pm, OUTPUT);
  pinMode(D6_Dp, OUTPUT);
  pinMode(D6_Dm, OUTPUT);
  //
  pinMode(D7_Pp, OUTPUT);
  pinMode(D7_Pm, OUTPUT);
  pinMode(D7_Dp, OUTPUT);
  pinMode(D7_Dm, OUTPUT);
  //
  pinMode(D8_Pp, OUTPUT);
  pinMode(D8_Pm, OUTPUT);
  pinMode(D8_Dp, OUTPUT);
  pinMode(D8_Dm, OUTPUT);
  //

  // Do we need to explicitly set pinmodes? Other version on Arduino Uno didn't. 
  //pinMode(J1stepPin, OUTPUT);
  D1_Stepper.setSpeed(motorSpeed1);
  D2_Stepper.setSpeed(motorSpeed2);
  D3_Stepper.setSpeed(motorSpeed3);
  D4_Stepper.setSpeed(motorSpeed4);
  D5_Stepper.setSpeed(motorSpeed5);
  D6_Stepper.setSpeed(motorSpeed6);
  D7_Stepper.setSpeed(motorSpeed7);
  D8_Stepper.setSpeed(motorSpeed8);
  
}


void loop() {
  while (Serial.available() > 0)
  {
    char recieved = Serial.read();
    inData += recieved;
    // Process message when new line character is recieved
    if (recieved == '\n')
    {
      String function = inData.substring(0, 2);

      //----- MANUAL TESTING---------------------------------------------------
      //-----------------------------------------------------------------------
      
      if (function == "TT") // "Basic testing"
      {
        //Serial.print("=== "); Serial.print(inData); Serial.println();
        int J1start = inData.indexOf('A');
        int J2start = inData.indexOf('B');
        int J3start = inData.indexOf('C');
        int J4start = inData.indexOf('D');
        int J5start = inData.indexOf('E');
        int J6start = inData.indexOf('F');
        int J7start = inData.indexOf('G');
        int J8start = inData.indexOf('H');
        int SPstart = inData.indexOf('S');
        int J1dir = inData.substring(J1start + 1, J1start + 2).toInt(); if (J1dir == 0) J1dir = -1;
        int J2dir = inData.substring(J2start + 1, J2start + 2).toInt(); if (J2dir == 0) J2dir = -1;
        int J3dir = inData.substring(J3start + 1, J3start + 2).toInt(); if (J3dir == 0) J3dir = -1;
        int J4dir = inData.substring(J4start + 1, J4start + 2).toInt(); if (J4dir == 0) J4dir = -1;
        int J5dir = inData.substring(J5start + 1, J5start + 2).toInt(); if (J5dir == 0) J5dir = -1;
        int J6dir = inData.substring(J6start + 1, J6start + 2).toInt(); if (J6dir == 0) J6dir = -1;
        int J7dir = inData.substring(J7start + 1, J7start + 2).toInt(); if (J7dir == 0) J7dir = -1;
        int J8dir = inData.substring(J8start + 1, J8start + 2).toInt(); if (J8dir == 0) J8dir = -1;
        int J1step = inData.substring(J1start + 2, J2start).toInt();
        int J2step = inData.substring(J2start + 2, J3start).toInt();
        int J3step = inData.substring(J3start + 2, J4start).toInt();
        int J4step = inData.substring(J4start + 2, J5start).toInt();
        int J5step = inData.substring(J5start + 2, J6start).toInt();
        int J6step = inData.substring(J6start + 2, J7start).toInt();
        int J7step = inData.substring(J7start + 2, J8start).toInt();
        int J8step = inData.substring(J8start + 2, SPstart).toInt();
        float SpeedIn = atof(inData.substring(SPstart + 1).c_str()); //.toFloat();
        //SpeedIn = (SpeedIn / 100);
        //float CalcSpeed = (1600 / SpeedIn);
        //int Speed = int(CalcSpeed);
        
        // Short-term testing mode: make sure to call MJ with 1 command at a time, 
        // e.g. "TBA0400" otherwise will sequentially move axes in order
        
        if (J1step > 0) {
          D1_Stepper.step(J1dir * J1step); 
          Serial.print("Stepped D1: "); Serial.println(J1dir * J1step);
        }
        if (J2step > 0) {
          D2_Stepper.step(J2dir * J2step); 
          Serial.print("Stepped D2: "); Serial.println(J2dir * J2step);
        }
        if (J3step > 0) {
          D3_Stepper.step(J3dir * J3step); 
          Serial.print("Stepped D3: "); Serial.println(J3dir * J3step);
        }
        if (J4step > 0) {
          D4_Stepper.step(J4dir * J4step); 
          Serial.print("Stepped D4: "); Serial.println(J4dir * J4step);
        }
        if (J5step > 0) {
          D5_Stepper.step(J5dir * J5step); 
          Serial.print("Stepped D5: "); Serial.println(J5dir * J5step);
        }
        if (J6step > 0) {
          D6_Stepper.step(J6dir * J6step); 
          Serial.print("Stepped D6: "); Serial.println(J6dir * J6step);
        }
        if (J7step > 0) {
          D7_Stepper.step(J7dir * J7step); 
          Serial.print("Stepped D7: "); Serial.println(J7dir * J7step);
        }
        if (J8step > 0) {
          D8_Stepper.step(J8dir * J8step); 
          Serial.print("Stepped D8: "); Serial.println(J8dir * J8step);
        }
        
      }

      //-----COMMAND TO WAIT TIME---------------------------------------------------
      //-----------------------------------------------------------------------
      if (function == "WT")
      {
        int WTstart = inData.indexOf('S');
        float WaitTime = atof(inData.substring(WTstart + 1).c_str()); //inData.substring(WTstart + 1).toFloat();
        int WaitTimeMS = WaitTime * 1000;
        delay(WaitTimeMS);
        Serial.print("Done");
      }

      //-----COMMAND IF INPUT THEN JUMP---------------------------------------------------
      //-----------------------------------------------------------------------
      if (function == "IJ")
      {
        int IJstart = inData.indexOf('X');
        int IJTabstart = inData.indexOf('T');
        int IJInputNum = inData.substring(IJstart + 1, IJTabstart).toInt();
        if (digitalRead(IJInputNum) == HIGH)
        {
          Serial.println("True\n");
        }
        if (digitalRead(IJInputNum) == LOW)
        {
          Serial.println("False\n");
        }
      }
      //-----COMMAND SET OUTPUT ON---------------------------------------------------
      //-----------------------------------------------------------------------
      if (function == "ON")
      {
        int ONstart = inData.indexOf('X');
        int outputNum = inData.substring(ONstart + 1).toInt();
        digitalWrite(outputNum, HIGH);
        Serial.print("Done");
      }
      //-----COMMAND SET OUTPUT OFF---------------------------------------------------
      //-----------------------------------------------------------------------
      if (function == "OF")
      {
        int ONstart = inData.indexOf('X');
        int outputNum = inData.substring(ONstart + 1).toInt();
        digitalWrite(outputNum, LOW);
        Serial.print("Done");
      }
      //-----COMMAND TO WAIT INPUT ON---------------------------------------------------
      //-----------------------------------------------------------------------
      if (function == "WI")
      {
        int WIstart = inData.indexOf('N');
        int InputNum = inData.substring(WIstart + 1).toInt();
        while (digitalRead(InputNum) == LOW) {
          delay(100);
        }
        Serial.print("Done");
      }
      //-----COMMAND TO WAIT INPUT OFF---------------------------------------------------
      //-----------------------------------------------------------------------
      if (function == "WO")
      {
        int WIstart = inData.indexOf('N');
        int InputNum = inData.substring(WIstart + 1).toInt();

        //String InputStr =  String("Input" + InputNum);
        //uint8_t Input = atoi(InputStr.c_str ());
        while (digitalRead(InputNum) == HIGH) {
          delay(100);
        }
        Serial.print("Done");
      }
      
      
      //-----COMMAND TO SET SPEEDS --------------------------------------------
      //-----------------------------------------------------------------------
      if (function == "SS")
      {
        //Serial.print("=== "); Serial.print(inData); Serial.println();
        int J1start = inData.indexOf('A');
        int J2start = inData.indexOf('B');
        int J3start = inData.indexOf('C');
        int J4start = inData.indexOf('D');
        int J5start = inData.indexOf('E');
        int J6start = inData.indexOf('F');
        int J7start = inData.indexOf('G');
        int J8start = inData.indexOf('H');
        int J1speed = inData.substring(J1start + 1, J2start).toInt(); 
        int J2speed = inData.substring(J2start + 1, J3start).toInt(); 
        int J3speed = inData.substring(J3start + 1, J4start).toInt(); 
        int J4speed = inData.substring(J4start + 1, J5start).toInt(); 
        int J5speed = inData.substring(J5start + 1, J6start).toInt(); 
        int J6speed = inData.substring(J6start + 1, J7start).toInt(); 
        int J7speed = inData.substring(J7start + 1, J8start).toInt(); 
        int J8speed = inData.substring(J8start + 1).toInt(); 
        
        if (J1speed > 0 && J1speed < SPEED_MAX) {
            motorSpeed1 = J1speed;
            D1_Stepper.setSpeed(motorSpeed1);
            //Serial.println("Set driver 1 to speed: "); Serial.println(motorSpeed1);
        }
        if (J2speed > 0 && J2speed < SPEED_MAX) {
            motorSpeed2 = J2speed;
            D2_Stepper.setSpeed(motorSpeed2);
            //Serial.println("Set driver 2 to speed: "); Serial.println(motorSpeed2);
        }
        if (J3speed > 0 && J3speed < SPEED_MAX) {
            motorSpeed3 = J3speed;
            D3_Stepper.setSpeed(motorSpeed3);
            //Serial.println("Set driver 3 to speed: "); Serial.println(motorSpeed3);
        }
        if (J4speed > 0 && J4speed < SPEED_MAX) {
            motorSpeed4 = J4speed;
            D4_Stepper.setSpeed(motorSpeed4);
            //Serial.println("Set driver 4 to speed: "); Serial.println(motorSpeed4);
        }
        if (J5speed > 0 && J5speed < SPEED_MAX) {
            motorSpeed5 = J5speed;
            D5_Stepper.setSpeed(motorSpeed5);
            //Serial.println("Set driver 5 to speed: "); Serial.println(motorSpeed5);
        }
        if (J6speed > 0 && J6speed < SPEED_MAX) {
            motorSpeed6 = J6speed;
            D6_Stepper.setSpeed(motorSpeed6);
            //Serial.println("Set driver 6 to speed: "); Serial.println(motorSpeed6);
        }
        if (J7speed > 0 && J7speed < SPEED_MAX) {
            motorSpeed7 = J7speed;
            D7_Stepper.setSpeed(motorSpeed7);
            //Serial.println("Set driver 7 to speed: "); Serial.println(motorSpeed7);
        }
        if (J8speed > 0 && J8speed < SPEED_MAX) {
            motorSpeed8 = J8speed;
            D8_Stepper.setSpeed(motorSpeed8);
            //Serial.println("Set driver 8 to speed: "); Serial.println(motorSpeed8);
        }

        // Update step delays: 
        stepDelay1 = 60.0 * 1000.0 / stepsPerRevolution1 * 1000.0 / motorSpeed1;
        stepDelay2 = 60.0 * 1000.0 / stepsPerRevolution2 * 1000.0 / motorSpeed2;
        stepDelay3 = 60.0 * 1000.0 / stepsPerRevolution3 * 1000.0 / motorSpeed3;
        stepDelay4 = 60.0 * 1000.0 / stepsPerRevolution4 * 1000.0 / motorSpeed4;
        stepDelay5 = 60.0 * 1000.0 / stepsPerRevolution5 * 1000.0 / motorSpeed5;
        stepDelay6 = 60.0 * 1000.0 / stepsPerRevolution6 * 1000.0 / motorSpeed6;
        stepDelay7 = 60.0 * 1000.0 / stepsPerRevolution7 * 1000.0 / motorSpeed7;
        stepDelay8 = 60.0 * 1000.0 / stepsPerRevolution8 * 1000.0 / motorSpeed8;
        
        /*
         * Serial.println("motorSpeed1: "); Serial.println(motorSpeed1);
        Serial.println("motorSpeed2: "); Serial.println(motorSpeed2);
        Serial.println("motorSpeed3: "); Serial.println(motorSpeed3);
        Serial.println("motorSpeed4: "); Serial.println(motorSpeed4);
        Serial.println("motorSpeed5: "); Serial.println(motorSpeed5);
        Serial.println("motorSpeed6: "); Serial.println(motorSpeed6);
        Serial.println("motorSpeed7: "); Serial.println(motorSpeed7);
        Serial.println("motorSpeed8: "); Serial.println(motorSpeed8);
        Serial.println("stepDelay1: "); Serial.println(stepDelay1);
        Serial.println("stepDelay2: "); Serial.println(stepDelay2);
        Serial.println("stepDelay3: "); Serial.println(stepDelay3);
        Serial.println("stepDelay4: "); Serial.println(stepDelay4);
        Serial.println("stepDelay5: "); Serial.println(stepDelay5);
        Serial.println("stepDelay6: "); Serial.println(stepDelay6);
        Serial.println("stepDelay7: "); Serial.println(stepDelay7);
        Serial.println("stepDelay8: "); Serial.println(stepDelay8);
        */
        Serial.println("Set speed successfully.");
        
      }
      
      
      
      //-----COMMAND TO MOVE---------------------------------------------------
      //-----------------------------------------------------------------------
      if (function == "MJ")
      {
        //Serial.print("=== "); Serial.print(inData); Serial.println();
        int J1start = inData.indexOf('A');
        int J2start = inData.indexOf('B');
        int J3start = inData.indexOf('C');
        int J4start = inData.indexOf('D');
        int J5start = inData.indexOf('E');
        int J6start = inData.indexOf('F');
        int J7start = inData.indexOf('G');
        int J8start = inData.indexOf('H');
        int MMstart = inData.indexOf('Z');
        int J1dir = inData.substring(J1start + 1, J1start + 2).toInt(); if (J1dir == 0) J1dir = -1;
        int J2dir = inData.substring(J2start + 1, J2start + 2).toInt(); if (J2dir == 0) J2dir = -1;
        int J3dir = inData.substring(J3start + 1, J3start + 2).toInt(); if (J3dir == 0) J3dir = -1;
        int J4dir = inData.substring(J4start + 1, J4start + 2).toInt(); if (J4dir == 0) J4dir = -1;
        int J5dir = inData.substring(J5start + 1, J5start + 2).toInt(); if (J5dir == 0) J5dir = -1;
        int J6dir = inData.substring(J6start + 1, J6start + 2).toInt(); if (J6dir == 0) J6dir = -1;
        int J7dir = inData.substring(J7start + 1, J7start + 2).toInt(); if (J7dir == 0) J7dir = -1;
        int J8dir = inData.substring(J8start + 1, J8start + 2).toInt(); if (J8dir == 0) J8dir = -1;
        float J1step = inData.substring(J1start + 2, J2start).toInt();
        float J2step = inData.substring(J2start + 2, J3start).toInt();
        float J3step = inData.substring(J3start + 2, J4start).toInt();
        float J4step = inData.substring(J4start + 2, J5start).toInt();
        float J5step = inData.substring(J5start + 2, J6start).toInt();
        float J6step = inData.substring(J6start + 2, J7start).toInt();
        float J7step = inData.substring(J7start + 2, J8start).toInt();
        float J8step = inData.substring(J8start + 2, MMstart).toInt();
        int motion_mode = 0;
        motion_mode = inData.substring(MMstart + 1).toInt();
        
        // Update step delays: 
        stepDelay1 = 60.0 * 1000.0 / stepsPerRevolution1 * 1000.0 / motorSpeed1;
        stepDelay2 = 60.0 * 1000.0 / stepsPerRevolution2 * 1000.0 / motorSpeed2;
        stepDelay3 = 60.0 * 1000.0 / stepsPerRevolution3 * 1000.0 / motorSpeed3;
        stepDelay4 = 60.0 * 1000.0 / stepsPerRevolution4 * 1000.0 / motorSpeed4;
        stepDelay5 = 60.0 * 1000.0 / stepsPerRevolution5 * 1000.0 / motorSpeed5;
        stepDelay6 = 60.0 * 1000.0 / stepsPerRevolution6 * 1000.0 / motorSpeed6;
        stepDelay7 = 60.0 * 1000.0 / stepsPerRevolution7 * 1000.0 / motorSpeed7;
        stepDelay8 = 60.0 * 1000.0 / stepsPerRevolution8 * 1000.0 / motorSpeed8;
        int maxStepDelay = max(stepDelay1, stepDelay2);
        maxStepDelay = max(maxStepDelay, stepDelay3);
        maxStepDelay = max(maxStepDelay, stepDelay4);
        maxStepDelay = max(maxStepDelay, stepDelay5);
        maxStepDelay = max(maxStepDelay, stepDelay6);
        maxStepDelay = max(maxStepDelay, stepDelay7);
        maxStepDelay = max(maxStepDelay, stepDelay8);
        float maxStepCount = max(J1step, J2step);
        maxStepCount = max(maxStepCount, J3step);
        maxStepCount = max(maxStepCount, J4step);
        maxStepCount = max(maxStepCount, J5step);
        maxStepCount = max(maxStepCount, J6step);
        maxStepCount = max(maxStepCount, J7step);
        maxStepCount = max(maxStepCount, J8step);
        long last_time_stamp_1 = 0; 
        long last_time_stamp_2 = 0; 
        long last_time_stamp_3 = 0; 
        long last_time_stamp_4 = 0; 
        long last_time_stamp_5 = 0; 
        long last_time_stamp_6 = 0; 
        long last_time_stamp_7 = 0; 
        long last_time_stamp_8 = 0; 
        long curr_time_stamp = micros(); 


        //FIND HIGHEST STEP (TIME), i.e. num steps x time per step.
        long J1stepTime = J1step * stepDelay1; long highStepTime =  J1stepTime; 
        long J2stepTime = J2step * stepDelay2;
        long J3stepTime = J3step * stepDelay3;
        long J4stepTime = J4step * stepDelay4;
        long J5stepTime = J5step * stepDelay5;
        long J6stepTime = J6step * stepDelay6;
        long J7stepTime = J7step * stepDelay7;
        long J8stepTime = J8step * stepDelay8;
        if (J2stepTime > highStepTime) {
          highStepTime = J2stepTime;
        }
        if (J3stepTime > highStepTime) {
          highStepTime = J3stepTime;
        }
        if (J4stepTime > highStepTime) {
          highStepTime = J4stepTime;
        }
        if (J5stepTime > highStepTime) {
          highStepTime = J5stepTime;
        }
        if (J6stepTime > highStepTime) {
          highStepTime = J6stepTime;
        }
        if (J7stepTime > highStepTime) {
          highStepTime = J7stepTime;
        }
        if (J8stepTime > highStepTime) {
          highStepTime = J8stepTime;
        }

        // This motion mode is for simultaneous motion: 
        // Lower the speeds of all the joints such that they will finish at the same time, based on num_steps 
        if (motion_mode == 1) { 
          float speedAdjustFactor = 1; //0.8; ///0.8; // empirically this works - theoretically not sure why just yet.
          float tempSpeed1 = motorSpeed1; float tempSpeed2 = motorSpeed2; float tempSpeed3 = motorSpeed3; float tempSpeed4 = motorSpeed4;
          float tempSpeed5 = motorSpeed5; float tempSpeed6 = motorSpeed6; float tempSpeed7 = motorSpeed7; float tempSpeed8 = motorSpeed8;
          // Only reduce the speeds of the non-max step counts (keep the max step count joint the same speed) 
          if (maxStepCount > J1step) tempSpeed1 = tempSpeed1 * speedAdjustFactor * (J1step / maxStepCount);
          if (maxStepCount > J2step) tempSpeed2 = tempSpeed2 * speedAdjustFactor * (J2step / maxStepCount);
          if (maxStepCount > J3step) tempSpeed3 = tempSpeed3 * speedAdjustFactor * (J3step / maxStepCount);
          if (maxStepCount > J4step) tempSpeed4 = tempSpeed4 * speedAdjustFactor * (J4step / maxStepCount);
          if (maxStepCount > J5step) tempSpeed5 = tempSpeed5 * speedAdjustFactor * (J5step / maxStepCount);
          if (maxStepCount > J6step) tempSpeed6 = tempSpeed6 * speedAdjustFactor * (J6step / maxStepCount);
          if (maxStepCount > J7step) tempSpeed7 = tempSpeed7 * speedAdjustFactor * (J7step / maxStepCount);
          if (maxStepCount > J8step) tempSpeed8 = tempSpeed8 * speedAdjustFactor * (J8step / maxStepCount);

          stepDelay1 = 60.0 * 1000.0 / stepsPerRevolution1 * 1000.0 / tempSpeed1;
          stepDelay2 = 60.0 * 1000.0 / stepsPerRevolution2 * 1000.0 / tempSpeed2;
          stepDelay3 = 60.0 * 1000.0 / stepsPerRevolution3 * 1000.0 / tempSpeed3;
          stepDelay4 = 60.0 * 1000.0 / stepsPerRevolution4 * 1000.0 / tempSpeed4;
          stepDelay5 = 60.0 * 1000.0 / stepsPerRevolution5 * 1000.0 / tempSpeed5;
          stepDelay6 = 60.0 * 1000.0 / stepsPerRevolution6 * 1000.0 / tempSpeed6;
          stepDelay7 = 60.0 * 1000.0 / stepsPerRevolution7 * 1000.0 / tempSpeed7;
          stepDelay8 = 60.0 * 1000.0 / stepsPerRevolution8 * 1000.0 / tempSpeed8; 
          
          //Serial.println("Motion mode 1 activated.");
        }

        // FIND THE MIN DELAY (what happens if JXstep is 0 ? Then min will become 0
        long minDelay = stepDelay1; if (minDelay <= 0) minDelay = 60.0 * 1000.0 / 200.0 * 1000.0 / 400.0; // normalize to 400 rpm in case bad command given
        if (stepDelay2 < minDelay && stepDelay2 > 0) {
          minDelay = stepDelay2; 
        }
        if (stepDelay3 < minDelay && stepDelay3 > 0) {
          minDelay = stepDelay3; 
        }
        if (stepDelay4 < minDelay && stepDelay4 > 0) {
          minDelay = stepDelay4; 
        }
        if (stepDelay5 < minDelay && stepDelay5 > 0) {
          minDelay = stepDelay5; 
        }
        if (stepDelay6 < minDelay && stepDelay6 > 0) {
          minDelay = stepDelay6; 
        }
        if (stepDelay7 < minDelay && stepDelay7 > 0) {
          minDelay = stepDelay7; 
        }
        if (stepDelay8 < minDelay && stepDelay8 > 0) {
          minDelay = stepDelay8; 
        }
        if (minDelay < 0) {
          Serial.println("Error: minDelay less than zero. Reseting to 400rpm equivalent.");
          minDelay = 60.0 * 1000.0 / 200.0 * 1000.0 / 400.0; // normalize to 400 rpm in case bad command given
        }

        //DETERMINE AXIS SKIP INCREMENT (old code, commented out below)
        /*
        int J1skip = (highStepTime / J1stepTime);
        int J2skip = (highStepTime / J2stepTime);
        int J3skip = (highStepTime / J3stepTime);
        int J4skip = (highStepTime / J4stepTime);
        int J5skip = (highStepTime / J5stepTime);
        int J6skip = (highStepTime / J6stepTime);
        int J7skip = (highStepTime / J7stepTime);
        int J8skip = (highStepTime / J8stepTime);
        */
        
        //RESET COUNTERS
        int J1done = 0;
        int J2done = 0;
        int J3done = 0;
        int J4done = 0;
        int J5done = 0;
        int J6done = 0;
        int J7done = 0;
        int J8done = 0;

        //RESET SKIP CURRENT
        int J1skipCur = 0;
        int J2skipCur = 0;
        int J3skipCur = 0;
        int J4skipCur = 0;
        int J5skipCur = 0;
        int J6skipCur = 0;
        int J7skipCur = 0;
        int J8skipCur = 0;

        int didJ1step = 0;
        int didJ2step = 0;
        int didJ3step = 0;
        int didJ4step = 0;
        int didJ5step = 0;
        int didJ6step = 0;
        int didJ7step = 0;
        int didJ8step = 0;
        
  
        //SET DIRECTIONS
        if (J1dir == 1) {
          digitalWrite(J1dirPin, HIGH);
        } else if (J1dir == -1) {
          digitalWrite(J1dirPin, LOW);
        }
        if (J2dir == 1) {
          digitalWrite(J2dirPin, HIGH);
        } else if (J2dir == -1) {
          digitalWrite(J2dirPin, LOW);
        }
        if (J3dir == 1) {
          digitalWrite(J3dirPin, HIGH);
        } else if (J3dir == -1) {
          digitalWrite(J3dirPin, LOW);
        }
        if (J4dir == 1) {
          digitalWrite(J4dirPin, HIGH);
        } else if (J4dir == -1) {
          digitalWrite(J4dirPin, LOW);
        }
        if (J5dir == 1) {
          digitalWrite(J5dirPin, HIGH);
        } else if (J5dir == -1) {
          digitalWrite(J5dirPin, LOW);
        }
        if (J6dir == 1) {
          digitalWrite(J6dirPin, HIGH);
        } else if (J6dir == -1) {
          digitalWrite(J6dirPin, LOW);
        }
        if (J7dir == 1) {
          digitalWrite(J7dirPin, HIGH);
        } else if (J7dir == -1) {
          digitalWrite(J7dirPin, LOW);
        }
        if (J8dir == 1) {
          digitalWrite(J8dirPin, HIGH);
        } else if (J8dir == -1) {
          digitalWrite(J8dirPin, LOW);
        }
        delayMicroseconds((int)PulseWidth);
        
        ////////////////////// PRINT VARIABLES /////////////////////
        int print_debug = 0;
        if (print_debug == 1) {
          Serial.println("---------- v PRINT DEBUG v ----------");
          Serial.print("J1step: "); Serial.println(J1step);
          Serial.print("J2step: "); Serial.println(J2step);
          Serial.print("J3step: "); Serial.println(J3step);
          Serial.print("J4step: "); Serial.println(J4step);
          Serial.print("J5step: "); Serial.println(J5step);
          Serial.print("J6step: "); Serial.println(J6step);
          Serial.print("J7step: "); Serial.println(J7step);
          Serial.print("J8step: "); Serial.println(J8step);
          Serial.print("stepDelay1: "); Serial.println(stepDelay1);
          Serial.print("stepDelay2: "); Serial.println(stepDelay2);
          Serial.print("stepDelay3: "); Serial.println(stepDelay3);
          Serial.print("stepDelay4: "); Serial.println(stepDelay4);
          Serial.print("stepDelay5: "); Serial.println(stepDelay5);
          Serial.print("stepDelay6: "); Serial.println(stepDelay6);
          Serial.print("stepDelay7: "); Serial.println(stepDelay7);
          Serial.print("stepDelay8: "); Serial.println(stepDelay8);
          Serial.print("minDelay: "); Serial.println(minDelay);
          Serial.print("J1stepTime: "); Serial.println(J1stepTime);
          Serial.print("J2stepTime: "); Serial.println(J2stepTime);
          Serial.print("J3stepTime: "); Serial.println(J3stepTime);
          Serial.print("J4stepTime: "); Serial.println(J4stepTime);
          Serial.print("J5stepTime: "); Serial.println(J5stepTime);
          Serial.print("J6stepTime: "); Serial.println(J6stepTime);
          Serial.print("J7stepTime: "); Serial.println(J7stepTime);
          Serial.print("J8stepTime: "); Serial.println(J8stepTime);
          Serial.print("highStepTime: "); Serial.println(highStepTime);
          //Serial.print("J1skip: "); Serial.println(J1skip);
          //Serial.print("J2skip: "); Serial.println(J2skip);
          //Serial.print("J3skip: "); Serial.println(J3skip);
          //Serial.print("J4skip: "); Serial.println(J4skip);
          //Serial.print("J5skip: "); Serial.println(J5skip);
          //Serial.print("J6skip: "); Serial.println(J6skip);
          //Serial.print("J7skip: "); Serial.println(J7skip);
          //Serial.print("J8skip: "); Serial.println(J8skip);
          Serial.print("Motion Mode: "); Serial.println(motion_mode);
          Serial.println("---------- ^ PRINT DEBUG ^ ----------");

        }

        // EXPERIMENT:
        minDelay = PulseWidth;
        
        //DRIVE MOTORS
        while (J1done < J1step || J2done < J2step || J3done < J3step || J4done < J4step || J5done < J5step || J6done < J6step || J7done < J7step || J8done < J8step)
        {

          curr_time_stamp = micros(); 
          // There's a chance that further along motors (4, 5, 6, 7, ...) will get a less accurate
          // time reading, i.e. curr_time won't be as updated as it should be by the time that code executes. 
          // But realistically this will lower RPM by a very small fractional amount...  
          
          if (J1done < J1step && J1skipCur == 0 && (curr_time_stamp - last_time_stamp_1 >= stepDelay1)) {
            digitalWrite(J1stepPin, HIGH); didJ1step = 1;
            last_time_stamp_1 = curr_time_stamp; 
          }
          if (J2done < J2step && J2skipCur == 0 && (curr_time_stamp - last_time_stamp_2 >= stepDelay2)) {
            digitalWrite(J2stepPin, HIGH); didJ2step = 1;
            last_time_stamp_2 = curr_time_stamp; 
          }
          if (J3done < J3step && J3skipCur == 0 && (curr_time_stamp - last_time_stamp_3 >= stepDelay3)) {
            digitalWrite(J3stepPin, HIGH); didJ3step = 1;
            last_time_stamp_3 = curr_time_stamp; 
          }
          if (J4done < J4step && J4skipCur == 0 && (curr_time_stamp - last_time_stamp_4 >= stepDelay4)) {
            digitalWrite(J4stepPin, HIGH); didJ4step = 1;
            last_time_stamp_4 = curr_time_stamp; 
          }
          if (J5done < J5step && J5skipCur == 0 && (curr_time_stamp - last_time_stamp_5 >= stepDelay5)) {
            digitalWrite(J5stepPin, HIGH); didJ5step = 1;
            last_time_stamp_5 = curr_time_stamp; 
          }
          if (J6done < J6step && J6skipCur == 0 && (curr_time_stamp - last_time_stamp_6 >= stepDelay6)) {
            digitalWrite(J6stepPin, HIGH); didJ6step = 1;
            last_time_stamp_6 = curr_time_stamp; 
          }
          //
          if (J7done < J7step && J7skipCur == 0 && (curr_time_stamp - last_time_stamp_7 >= stepDelay7)) {
            digitalWrite(J7stepPin, HIGH); didJ7step = 1;
            last_time_stamp_7 = curr_time_stamp; 
          }
          if (J8done < J8step && J8skipCur == 0 && (curr_time_stamp - last_time_stamp_8 >= stepDelay8)) {
            digitalWrite(J8stepPin, HIGH); didJ8step = 1;
            last_time_stamp_8 = curr_time_stamp; 
          }
          
          ////////////////////// TODO: figure out how they're setting speeds, counting steps until finished. Modify for our stepper lib.
          
          //#############DELAY AND SET LOW
          delayMicroseconds((int)PulseWidth); // Minimum pulse width the driver needs. 
          
          
          digitalWrite(J1stepPin, LOW);
          if (didJ1step == 1) J1done = ++J1done; // 
          digitalWrite(J2stepPin, LOW);
          if (didJ2step == 1) J2done = ++J2done; //
          digitalWrite(J3stepPin, LOW);
          if (didJ3step == 1) J3done = ++J3done; //
          digitalWrite(J4stepPin, LOW);
          if (didJ4step == 1) J4done = ++J4done; //
          digitalWrite(J5stepPin, LOW);
          if (didJ5step == 1) J5done = ++J5done; //
          digitalWrite(J6stepPin, LOW);
          if (didJ6step == 1) J6done = ++J6done; //
          digitalWrite(J7stepPin, LOW);
          if (didJ7step == 1) J7done = ++J7done; //
          digitalWrite(J8stepPin, LOW);
          if (didJ8step == 1) J8done = ++J8done; //
          
          
          // DELAY FOR SPEEEEED SETTINGS
          //delayMicroseconds((int)minDelay); 

          didJ1step = 0; didJ2step = 0; didJ3step = 0; didJ4step = 0;
          didJ5step = 0; didJ6step = 0; didJ7step = 0; didJ8step = 0;
          
          /*
          if (J1done < J1step && J1skipCur == 0) {
            digitalWrite(J1stepPin, LOW);
            J1done = ++J1done;
          }
          if (J2done < J2step && J2skipCur == 0) {
            digitalWrite(J2stepPin, LOW);
            J2done = ++J2done;
          }
          if (J3done < J3step && J3skipCur == 0) {
            digitalWrite(J3stepPin, LOW);
            J3done = ++J3done;
          }
          if (J4done < J4step && J4skipCur == 0) {
            digitalWrite(J4stepPin, LOW);
            J4done = ++J4done;
          }
          if (J5done < J5step && J5skipCur == 0) {
            digitalWrite(J5stepPin, LOW);
            J5done = ++J5done;;
          }
          if (J6done < J6step && J6skipCur == 0) {
            digitalWrite(J6stepPin, LOW);
            J6done = ++J6done;
          }
          if (J7done < J7step && J7skipCur == 0) {
            digitalWrite(J7stepPin, LOW); Serial.print("yeah buddy");
            J7done = ++J7done;
          }
          if (J8done < J8step && J8skipCur == 0) {
            digitalWrite(J8stepPin, LOW);
            J8done = ++J8done;
          }
          //#############DELAY BEFORE RESTARTING LOOP AND SETTING HIGH AGAIN
          delayMicroseconds(minDelay); // forces the speed to be fastest?? 
          
          //increment skip count
          J1skipCur = ++J1skipCur;
          J2skipCur = ++J2skipCur;
          J3skipCur = ++J3skipCur;
          J4skipCur = ++J4skipCur;
          J5skipCur = ++J5skipCur;
          J6skipCur = ++J6skipCur;
          J7skipCur = ++J7skipCur;
          J8skipCur = ++J8skipCur;
          //if skiped enough times set back to zero
          if (J1skipCur == J1skip) {
            J1skipCur = 0;
          }
          if (J2skipCur == J2skip) {
            J2skipCur = 0;
          }
          if (J3skipCur == J3skip) {
            J3skipCur = 0;
          }
          if (J4skipCur == J4skip) {
            J4skipCur = 0;
          }
          if (J5skipCur == J5skip) {
            J5skipCur = 0;
          }
          if (J6skipCur == J6skip) {
            J6skipCur = 0;
          }
          if (J7skipCur == J7skip) {
            J7skipCur = 0;
          }
          if (J8skipCur == J8skip) {
            J8skipCur = 0;
          }
          */
          
        }
        
        
        
        inData = ""; // Clear recieved buffer
        Serial.write('1'); // This is to tell the controller that we are ready for a new command
        Serial.println();
        //Serial.print("Move Done");
      }
      else
      {
        inData = ""; // Clear recieved buffer
      }
    }
  }
}


