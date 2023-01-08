#include <TensorFlowLite_ESP32.h>
/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/


#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "main_functions.h"
#include "model.h"
#include "constants.h"
#include "output_handler.h"
#include <Arduino.h>
// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

constexpr int kTensorArenaSize = 30*1024;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace



/***************************************************************************
  This is a library for the AMG88xx GridEYE 8x8 IR camera

  This sketch tries to read the pixels from the sensor

  Designed specifically to work with the Adafruit AMG88 breakout
  ----> http://www.adafruit.com/products/3538

  These sensors use I2C to communicate. The device's I2C address is 0x69

  Adafruit invests time and resources providing this open source code,
  please support Adafruit andopen-source hardware by purchasing products
  from Adafruit!

  Written by Dean Miller for Adafruit Industries.
  BSD license, all text above must be included in any redistribution
 ***************************************************************************/
#include <SPI.h>
#include <Wire.h>
#include <Adafruit_AMG88xx.h>
#include <TFT_eSPI.h> // Graphics and font library for ILI9341 driver chip

#include "interpolation.h"

//low range of the sensor (this will be blue on the screen)
#define MINTEMP 0

//high range of the sensor (this will be red on the screen)
#define MAXTEMP 40

#define AMG_COLS 8
#define AMG_ROWS 8
#define INTERPOLATED_COLS 24
#define INTERPOLATED_ROWS 24

#define THERMO_INPUT_SIZE 576

TFT_eSPI tft = TFT_eSPI();  // Invoke library, pins defined in User_Setup.h
Adafruit_AMG88xx amg;

float pixels[AMG88xx_PIXEL_ARRAY_SIZE];
float thermo_input[THERMO_INPUT_SIZE];
uint16_t displayPixelWidth, displayPixelHeight;

//the colors we will be using
const uint16_t camColors[] = {0x480F,
0x400F,0x400F,0x400F,0x4010,0x3810,0x3810,0x3810,0x3810,0x3010,0x3010,
0x3010,0x2810,0x2810,0x2810,0x2810,0x2010,0x2010,0x2010,0x1810,0x1810,
0x1811,0x1811,0x1011,0x1011,0x1011,0x0811,0x0811,0x0811,0x0011,0x0011,
0x0011,0x0011,0x0011,0x0031,0x0031,0x0051,0x0072,0x0072,0x0092,0x00B2,
0x00B2,0x00D2,0x00F2,0x00F2,0x0112,0x0132,0x0152,0x0152,0x0172,0x0192,
0x0192,0x01B2,0x01D2,0x01F3,0x01F3,0x0213,0x0233,0x0253,0x0253,0x0273,
0x0293,0x02B3,0x02D3,0x02D3,0x02F3,0x0313,0x0333,0x0333,0x0353,0x0373,
0x0394,0x03B4,0x03D4,0x03D4,0x03F4,0x0414,0x0434,0x0454,0x0474,0x0474,
0x0494,0x04B4,0x04D4,0x04F4,0x0514,0x0534,0x0534,0x0554,0x0554,0x0574,
0x0574,0x0573,0x0573,0x0573,0x0572,0x0572,0x0572,0x0571,0x0591,0x0591,
0x0590,0x0590,0x058F,0x058F,0x058F,0x058E,0x05AE,0x05AE,0x05AD,0x05AD,
0x05AD,0x05AC,0x05AC,0x05AB,0x05CB,0x05CB,0x05CA,0x05CA,0x05CA,0x05C9,
0x05C9,0x05C8,0x05E8,0x05E8,0x05E7,0x05E7,0x05E6,0x05E6,0x05E6,0x05E5,
0x05E5,0x0604,0x0604,0x0604,0x0603,0x0603,0x0602,0x0602,0x0601,0x0621,
0x0621,0x0620,0x0620,0x0620,0x0620,0x0E20,0x0E20,0x0E40,0x1640,0x1640,
0x1E40,0x1E40,0x2640,0x2640,0x2E40,0x2E60,0x3660,0x3660,0x3E60,0x3E60,
0x3E60,0x4660,0x4660,0x4E60,0x4E80,0x5680,0x5680,0x5E80,0x5E80,0x6680,
0x6680,0x6E80,0x6EA0,0x76A0,0x76A0,0x7EA0,0x7EA0,0x86A0,0x86A0,0x8EA0,
0x8EC0,0x96C0,0x96C0,0x9EC0,0x9EC0,0xA6C0,0xAEC0,0xAEC0,0xB6E0,0xB6E0,
0xBEE0,0xBEE0,0xC6E0,0xC6E0,0xCEE0,0xCEE0,0xD6E0,0xD700,0xDF00,0xDEE0,
0xDEC0,0xDEA0,0xDE80,0xDE80,0xE660,0xE640,0xE620,0xE600,0xE5E0,0xE5C0,
0xE5A0,0xE580,0xE560,0xE540,0xE520,0xE500,0xE4E0,0xE4C0,0xE4A0,0xE480,
0xE460,0xEC40,0xEC20,0xEC00,0xEBE0,0xEBC0,0xEBA0,0xEB80,0xEB60,0xEB40,
0xEB20,0xEB00,0xEAE0,0xEAC0,0xEAA0,0xEA80,0xEA60,0xEA40,0xF220,0xF200,
0xF1E0,0xF1C0,0xF1A0,0xF180,0xF160,0xF140,0xF100,0xF0E0,0xF0C0,0xF0A0,
0xF080,0xF060,0xF040,0xF020,0xF800,};





void inference_input(float *p, uint8_t rows, uint8_t cols) 
{
  int i = 0;
  float val_limit;

  for (int y=0; y<rows; y++) 
  {
    for (int x=0; x<cols; x++) 
    {
      float val = get_point(p, rows, cols, x, y);
      if(val >= MAXTEMP) val_limit = MAXTEMP;
      else if(val <= MINTEMP) val_limit = MINTEMP;
      else val_limit = val;
      
      thermo_input[i++] = val_limit;

    }
  }

  float bias = 0;
  for(int i = 0;i<THERMO_INPUT_SIZE;i++)
  {
    bias += thermo_input[i]/THERMO_INPUT_SIZE;
  }

  for(int i = 0;i<THERMO_INPUT_SIZE;i++)
  {
    input->data.f[i] = (thermo_input[i]-bias)/10;
  }

}

typedef struct 
{
  int16_t x;
  int16_t y;
  int16_t h;
  int16_t w;
}Bar_t;

Bar_t left_bar,right_bar;

uint8_t opt = 0;
uint8_t last_opt = 0;

void bar_init(void)
{
    left_bar.h = 20;
    right_bar.h = 20;
    
    left_bar.w = 30;
    right_bar.w = 30;
    
    
    left_bar.x = 50;
    right_bar.x = 180;
    
    left_bar.y = 220;
    right_bar.y = 220;
}




void bar_show(int index)
{
  static uint8_t cnt;
  if(++cnt==12){
    cnt = 0;

  switch(index)
  {
    case 0:  opt = 0;
             break;

    case 2:  opt = 1;
             break;
    
    case 4:  opt = 2;
             break;

    case 1:  if(opt==1)
             {
                left_bar.h -= 40;
                left_bar.y += 40;
                if(left_bar.y>220 || left_bar.h<20)
                {
                  left_bar.y = 220;
                  left_bar.h = 20;
                }
             }
             else if(opt==2)
             {
                right_bar.h -= 40;
                right_bar.y += 40;
                if(right_bar.y>220 || right_bar.h<20)
                {
                  right_bar.y = 220;
                  right_bar.h = 20;
                }
             }
             break; 

    case 3:  if(opt==1)
             {
                left_bar.h += 40;
                left_bar.y -= 40;
                if(left_bar.h>220 || left_bar.y<20)
                {
                  left_bar.h = 220;
                  left_bar.y = 20;
                }
             }
             else if(opt==2)
             {
                right_bar.h += 40;
                right_bar.y -= 40;
                if(right_bar.h>220 || right_bar.y<20)
                {
                  right_bar.h = 220;
                  right_bar.y = 20;
                }
             }
             break; 

    default: break;
  }
  

  if(last_opt!=opt)
  {
    if(opt==0)
    {
      left_bar.w = 30;
      right_bar.w = 30;
    }
    else if(opt==1)
    {
      left_bar.w = 50;
      right_bar.w = 30;
    }    
    else if(opt==2)
    {
      left_bar.w = 30;
      right_bar.w = 50;
    }
    last_opt = opt;

    tft.fillRect(left_bar.x,left_bar.y,50,left_bar.h,TFT_BLACK);
    tft.fillRect(right_bar.x,right_bar.y,50,right_bar.h,TFT_BLACK);
  }


    tft.fillRect(left_bar.x,left_bar.y,left_bar.w,left_bar.h,TFT_RED);
    tft.fillRect(right_bar.x,right_bar.y,right_bar.w,right_bar.h,TFT_GREEN); 
    tft.fillRect(left_bar.x,0,left_bar.w,240-left_bar.h,TFT_BLACK);
    tft.fillRect(right_bar.x,0,right_bar.w,240-right_bar.h,TFT_BLACK);

    char buf[10];
    sprintf(buf,"%d",index);
    tft.drawString(buf, 0, 50, 4);

  }
}




void setup() 
{
    bar_init();

    tft.init();
    tft.setRotation(3);
    tft.fillScreen(TFT_BLACK);
    
    displayPixelWidth = tft.width() / 8;
    displayPixelHeight = tft.height() / 8;

   
    Serial.begin(921600);
 

    bool status;
    
    // default settings
    status = amg.begin();
    if (!status) {
        Serial.println("Could not find a valid AMG88xx sensor, check wiring!");
        while (1);
    }
  
    delay(100); // let sensor boot up

    



    

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

}


uint16_t cnt = 0;

void loop() 
{ 
  amg.readPixels(pixels);
    
    
    // for(int i=0; i<AMG88xx_PIXEL_ARRAY_SIZE; i++)
    // {
    //   uint8_t colorIndex = map(pixels[i], MINTEMP, MAXTEMP, 0, 255);
    //   colorIndex = (uint8_t)constrain((int16_t)colorIndex, (int16_t)0, (int16_t)255);
    //   th_input[i] = colorIndex;

    //   tft.fillRect(displayPixelHeight * floor(i / 8), displayPixelWidth * (i % 8),displayPixelHeight, displayPixelWidth, camColors[colorIndex]);
    // }
    // Serial.print(millis()-t); 
    // Serial.println("ms");

  // int32_t t = millis();
  
  float dest_2d[INTERPOLATED_ROWS * INTERPOLATED_COLS];
  interpolate_image(pixels, AMG_ROWS, AMG_COLS, dest_2d, INTERPOLATED_ROWS, INTERPOLATED_COLS);
  uint16_t boxsize = min(tft.width() / INTERPOLATED_COLS, tft.height() / INTERPOLATED_COLS);

  inference_input(dest_2d, INTERPOLATED_ROWS, INTERPOLATED_COLS);

  
  

  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed on x: %f\n",
                         static_cast<double>(0));
    return;
  }
  // Serial.println(millis()-t);



  float max = output->data.f[0];
  uint8_t index = 0;
  for(int i = 1;i<5;i++)
  {
    if(output->data.f[i]>max)
      {
        max = output->data.f[i];
        index = i;
      }
    
  }



  bar_show(index);


  // for(int i = 0;i<5;i++)
  // {
  //   Serial.print(output->data.f[i]);
  //   Serial.print(",");
  // }
  //   Serial.println();

}




