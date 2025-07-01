# TFLite_HSI_CNN

Repository for TFLite HSI CNN classification on the RP2040 microcontroller (Raspberry Pi Pico).

---

## 🛠️ Required Software

- **Arduino IDE** version 2.1.0 or later  
- **Web browser** with internet access (for Google Colab)

---

## 🧪 How to Run the Python Part

1. Open [Google Colab](https://colab.research.google.com/)
2. Load the notebook: `HSI_Training_2D.ipynb`
3. Run the **"Initialization"** section
4. Grant Google Colab access to your Google Drive
5. In the **"Global"** section, set the variables for execution
6. Run the remaining sections

### 📁 Output Files (inside `/Models` folder)

- `db_CNN.h` (where `db` depends on the selected dataset)
- `tensor_correct.h`
- `tensor_input.h`

---

## ⚙️ Arduino IDE Setup (First-Time Configuration)

1. Open **Arduino IDE**
2. Connect the **Raspberry Pi Pico**
3. In the left panel, go to **Boards Manager** (second icon)
4. Search for `Arduino Mbed RP2040 Boards` and install version **4.0.2 or newer**
5. Add TensorFlow Lite library:  
   Go to **Sketch → Include Library → Add .ZIP Library**, and load:  
   `Libs/Arduino_TensorFlowLite-2.4.0-ALPHA.zip`

---

## 🚀 Inference on the Board

1. Load the sketch:  
   `Sketch/sketch_cnn_in_i8_out_i8_db`
2. In the same folder as the sketch, add the previously generated data files
3. Click **"Upload"** to flash the board
4. To monitor execution:  
   Go to **Tools → Serial Monitor**


## Cite this:

J. V. S. Hütner, F. Viel, C. A. Zeferino and E. A. Bezerra, "TinyML Applied in Hyperspectral Image Classification on COTS Microcontroller," 2024 XIV Brazilian Symposium on Computing Systems Engineering (SBESC), Recife, Brazil, 2024, pp. 1-6, doi: 10.1109/SBESC65055.2024.10771925.

   @INPROCEEDINGS{hutner2024tinyML,
     author={Hütner, João Victor Santos and Viel, Felipe and Zeferino, Cesar A. and Bezerra, Eduardo Augusto},
     booktitle={2024 XIV Brazilian Symposium on Computing Systems Engineering (SBESC)},
     title={TinyML Applied in Hyperspectral Image Classification on COTS Microcontroller},
     year={2024},
     volume={},
     number={},
     pages={1-6},
     keywords={Deep learning;Adaptation models;Satellites;Microcontrollers;Tiny machine learning;Computational modeling;Convolutional neural networks;CubeSat;Hyperspectral imaging;Image classification;Hyperspectral Image;TinyML;CNN;CubeSat;COTS},
     doi={10.1109/SBESC65055.2024.10771925}}

