# TFLite_HSI_CNN
Repository of TFLite HSI CNN classification for RP2040 microcontroller (Raspberry Pi Pico).

Programas necessários:
- Arduino IDE 2.1.0 ou mais recente
- Qualquer Browser com conexão com a internet para acesso ao Google Colab

Como executar parte do Python:
- Acessar Google Colab
- Carregar HSI_Training_2D.ipynb
- Executar seção 'Initialization'
- Liberar acesso ao Driver para o Colab
- Em 'Global' configurar variáveis para execução
- Execute o restante das seções

Os seguintes dados serão gerados dentro do /Models:
- db_CNN.h (db - dependendo da base escolhida)
- tensor_correct.h
- tensor_input.h

Após gerar os dados, para configurar o Arduino IDE pela primeira vez:
- Acessar o Arduino IDE
- Conectar o Raspberry Pi Pico
- Na lateral esquerda, acessar Boards Manager (segundo ícone) e procurar "Arduino Mbed RP2040 Boards". Instalar a versão 4.0.2 ou mais recente.
- Carregar 'Libs/Arduino_TensorFlowLite-2.4.0-ALPHA.zip' (Sketch->Include Library->Add .Zip Library)

Após configurar, para inferenciar:
- Carregar Sketch/sketch_cnn_in_i8_out_i8_db
- No mesmo local que a sketch foi carregada, adicionar os dados gerados anteriormente.
- Clicar em "Upload" para enviar para a placa.
- Para observar a execução basta clicar em Tools->Serial Monitor
