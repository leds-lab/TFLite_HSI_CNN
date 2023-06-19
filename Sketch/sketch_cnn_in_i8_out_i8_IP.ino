#include "IP_CNN.h"
#include <TensorFlowLite.h>
#include "tensor_input.h"
//#include "tensor_output.h"
#include "tensor_correct.h"
#include <String.h>
/*
*
* Raspberry Pico CNN Classification
* Joao Hutner
* 01/05/2023
* 
*/

/*
-> Sites úteis:
https://github.com/tensorflow/tflite-micro-arduino-examples
https://www.tensorflow.org/lite/microcontrollers/get_started_low_level?hl=pt-br
https://raspberry-projects.com/pi/programming-in-c/memory/variables
https://repositorio.utfpr.edu.br/jspui/bitstream/1/30638/1/inteligenciaartificialembarcadostinyml.pdf

-> Sequencia para rodar:
1. Loading and parsing the model: TFLu parses the weights and network architecture stored in the C-byte array.
2. Transforming the input data: The input data acquired from the sensor is converted to the expected format required by the model.
3. Executing the model: TFLu executes the model using optimized DNN functions.

-> Problemas:
Didn't find op for builtin opcode 'EXPAND_DIMS' version '1'. An older version of this builtin might be supported. Are you using an old TFLite binary with a newer model?
Failed to get registration from op code EXPAND_DIMS

Aparentemente o Conv1D não é suportado pelo TensorFlowLite:
https://github.com/eloquentarduino/EloquentTinyML/issues/16

https://github.com/tensorflow/tflite-micro/issues/1157
https://stackoverflow.com/questions/66664484/tensorflow-lite-on-arduino-nano-33-ble-didnt-find-op-for-builtin-opcode-expa
https://github.com/rkuo2000/Arduino/tree/master/examples/TinyML
https://www.kaggle.com/code/rkuo2000/tinyml-ecg/notebook
*/

//Executing the model: TFLu executes the model using optimized DNN functions.
#include <tensorflow/lite/micro/all_ops_resolver.h>       //fornece as operações usadas pelo interpretador para executar o modelo.
#include <tensorflow/lite/micro/micro_error_reporter.h>   //gera informações de depuração.
#include <tensorflow/lite/micro/micro_interpreter.h>      //contém código para carregar e executar modelos.
#include <tensorflow/lite/schema/schema_generated.h>      //contém o esquema para o formato de arquivo de modelo do TensorFlow Lite FlatBuffer .
#include <tensorflow/lite/version.h>                      //fornece informações de versão para o esquema do TensorFlow Lite.
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>

#define NUM_BANDS 30
#define NUM_CLASS 16
#define NUM_DATA 769
#define NUM_MAX 769
#define DB IP_CNN_tflite

#define VAR_IN int8_t
#define VAR_OUT int8_t

/*typedef enum {
  kTfLiteNoType = 0,
  kTfLiteFloat32 = 1,
  kTfLiteInt32 = 2,
  kTfLiteUInt8 = 3,
  kTfLiteInt64 = 4,
  kTfLiteString = 5,
  kTfLiteBool = 6,
  kTfLiteInt16 = 7,
  kTfLiteComplex64 = 8,
  kTfLiteInt8 = 9,
} TfLiteType; */

//Declare the variables required by TFLu:
const tflite::Model* tflu_model = nullptr;            //The model parsed by the TFLu parser.
tflite::MicroInterpreter* tflu_interpreter = nullptr; //The pointer to TFLu interpreter.
TfLiteTensor* tflu_i_tensor = nullptr;                //The pointer to the model's input tensor.
TfLiteTensor* tflu_o_tensor = nullptr;                //The pointer to the model's output tensor.
tflite::MicroErrorReporter tflu_error;                //The model errors

//The arena's size depends on the model and is only determined by experiments
// 4 * 1024
constexpr int tensor_arena_size = 8 * 1024;           //Memory required by the TFLu interpreter. TFLu does not use dynamic allocation.
byte tensor_arena[tensor_arena_size] __attribute__((aligned(16)));


//Init variables of quantization
float tflu_i_scale = 0.0f;
int32_t tflu_i_zero_point = 0;
float tflu_o_scale = 0.0f;
int32_t tflu_o_zero_point = 0;

//Posição do iterador (para dados)
int current_position = 0;

//Soma das corretas
int correct_predict = 0;

//Tempo Inicial e Final
int start_time = 0;
int end_time = 0;

//Inicialização do TensorFlowLite
void tflu_initialization()
{
  Serial.println("->TFLu initialization - Start");
  start_time = millis();

  // Load the TFLite model from the C-byte hsi_model_tflite
  tflu_model = tflite::GetModel(DB);
  //tflu_model = tflite::GetModel(IP_CNN_tflite);
  if (tflu_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print(tflu_model->version());
    Serial.println("");
    Serial.print(TFLITE_SCHEMA_VERSION);
    Serial.println("");
    Serial.print("Model provided schema not equal to supported version.");
    Serial.println("");
    while(1);
  }

  // Define a tflite::AllOpsResolver object
  // The TFLu interpreter will use this interface to find the function pointers for each DNN operator.
  tflite::AllOpsResolver tflu_ops_resolver;

  // Create the TFLu interpreter
  tflu_interpreter = new tflite::MicroInterpreter(tflu_model, tflu_ops_resolver, tensor_arena, tensor_arena_size, &tflu_error);

  // Allocate TFLu internal memory
  //tflu_interpreter->AllocateTensors();
  TfLiteStatus allocate_status = tflu_interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("");
    Serial.println("AllocateTensors() failed");
    Serial.println("");
    while(1);
  }
  

  //INPUT
  // Get the pointers for the input and output tensors
  tflu_i_tensor = tflu_interpreter->input(0);
  int in_size = tflu_i_tensor->dims->size;
  Serial.print("\nModel_in dimentions size: ");
  Serial.print(in_size, DEC);
  int i = 0;
  Serial.print("\nModel_in shape: ");
  for (i = 0; i < in_size; i++) {
    int dim = tflu_i_tensor->dims->data[i];
    Serial.print("[");
    Serial.print(dim, DEC);
    Serial.print("]");
  }
  Serial.print("\nModel_in type: ");
  Serial.print(tflu_i_tensor->type);
  Serial.print("\nModel_in bytes: ");
  Serial.print(tflu_i_tensor->bytes, DEC);

  //OUTPUT
  tflu_o_tensor = tflu_interpreter->output(0);
  if((tflu_o_tensor->dims->size != 2) || (tflu_o_tensor->dims->data[0] != 1) ||(tflu_o_tensor->dims->data[1] != NUM_CLASS)){
    Serial.println("\nBad output tensor parameters in model");
    while(1);
  }
  int out_size = tflu_o_tensor->dims->size;
  Serial.print("\nModel_out dimentions size: ");
  Serial.print(out_size, DEC);
  Serial.print("\nModel_out shape: ");
  for (i = 0; i < out_size; ++i) {
    int dim = tflu_o_tensor->dims->data[i];
    Serial.print("[");
    Serial.print(dim, DEC);
    Serial.print("]");
  }
  Serial.print("\nModel_out type: ");
  Serial.print(tflu_o_tensor->type);
  Serial.print("\nModel_out bytes: ");
  Serial.print(tflu_o_tensor->bytes, DEC);

  //PARAMETROS DE QUANT:
  
  const auto* i_quantization = reinterpret_cast<TfLiteAffineQuantization*>(tflu_i_tensor->quantization.params);
  const auto* o_quantization = reinterpret_cast<TfLiteAffineQuantization*>(tflu_o_tensor->quantization.params);
  tflu_i_scale = i_quantization->scale->data[0];
  tflu_i_zero_point = i_quantization->zero_point->data[0];
  tflu_o_scale = o_quantization->scale->data[0];
  tflu_o_zero_point = o_quantization->zero_point->data[0];
  Serial.print("\ntflu_i_scale: ");
  Serial.print(tflu_i_scale, DEC);
  Serial.print("\ntflu_i_zero_point: ");
  Serial.print(tflu_i_zero_point, DEC);
  Serial.print("\ntflu_o_scale: ");
  Serial.print(tflu_o_scale, DEC);
  Serial.print("\ntflu_o_zero_point: ");
  Serial.print(tflu_o_zero_point, DEC);

  Serial.print("\n\n->TFLu initialization - Completed - Execution time(ms): ");
  end_time = millis() - start_time;
  Serial.print(end_time, DEC);
  Serial.print("ms");
  Serial.print("\n\n");
}

void setup() {
  Serial.begin(9600);
  while (!Serial);
  tflu_initialization(); //Initialize TFLU
  Serial.print("\n\n->TFLu prediction - Start\n");
  start_time = millis();
}

/*
* MAIN LOOP
*/
void loop() {
  if(current_position > NUM_MAX-1){
    Serial.print("\n\n->TFLu prediction - Completed - Execution time(ms): ");
    end_time = millis() - start_time;
    Serial.print(end_time, DEC);
    Serial.print("ms");

    Serial.print("\nCorrect predictions: ");
    Serial.print(correct_predict, DEC);
    Serial.print("\nOverall Accuracy: ");
    float acc = 0.0;
    acc = (float(correct_predict)/float(NUM_MAX))*100.0;
    Serial.print(acc, DEC);
    Serial.print("%");

    while(1);
  }
  else{
    int i = 0;
    //Get input data
    //Serial.print("Current Position: ");
    //Serial.print(current_position, DEC);

    // ----> INPUT FROM .H
    //Serial.print("\nInput:\n");
    VAR_IN actual_sample[NUM_BANDS] = {};
    for (i = 0; i < NUM_BANDS; i++){
      actual_sample[i] = tensor_input[current_position][i];
      //Serial.print("[");
      //Serial.print(actual_sample[i], DEC);
      //Serial.print("]");
    }

    // Get a pointer to the input tensor data buffer
    VAR_IN* input_data = tflu_i_tensor->data.int8;
    
    // ----> INPUT -> FEED TENSOR
    //input_data[0] = {};
    for (i = 0; i < NUM_BANDS; i++){
      input_data[i] = actual_sample[i];
    }

    /*
    //Print input tensor
    Serial.print("\nInput tensor:\n");
    for (i = 0; i < NUM_BANDS; i++){
      Serial.print("[");
      Serial.print(input_data[i],DEC);
      Serial.print("]");
    }
    */

    // ----> RUN INFERENCE
    TfLiteStatus invoke_status = tflu_interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      Serial.println("Error invoking the TFLu interpreter");
      while(1);
    }

    // Get a pointer to the output tensor data buffer
    VAR_OUT* output_data = tflu_o_tensor->data.int8;
    // ----> OUTPUT
    int8_t max_score_i = 0;
    int max_index_i = 0;
    //Serial.print("\nPesos em Int:\n");
    for (i = 0; i < NUM_CLASS; i++) {
      VAR_OUT score = output_data[i];
      //Serial.print("[");
      //Serial.print(score, DEC);
      //Serial.print("]");
      if ((i == 0) || (score > max_score_i)) {
        max_score_i = score;
        max_index_i = i;
      }
    }
    int8_t saida_real = 0;
    saida_real = tensor_correct[current_position][0];

    /*
    //Print output tensor
    Serial.print("\nResultado Int: ");
    Serial.print(max_index_i, DEC);

    Serial.print("\nResultado Real: ");
    Serial.print(saida_real, DEC);

    //Next loop delay
    //Go to next data
    Serial.print("\n--------\n");
    */

    if(saida_real==max_index_i){
      correct_predict = correct_predict + 1;
    }
    current_position = current_position + 1;
  }
  return;
}