# CROWD_COUNTING_DEEPLEARNING
# 👥 Crowd Counting Application using Deep Learning

This project implements a **Crowd Counting System** using **MobileNetV2** as a backbone for feature extraction, combined with a **density map regression approach**. The model is trained on crowd images and learns to estimate the number of people using **density maps** rather than direct object detection. The system is memory-efficient and supports real-time inference via a **Streamlit** web interface.

---

## 📌 Key Features

- 📷 Upload an image and estimate how many people are present
- 📊 Visualize the **density map** using color gradients (Jet colormap)
- ⚡ Efficient model architecture using **MobileNetV2** + transposed convolutions
- 💾 Trained model checkpoint saved (`best_model.h5`) using Keras Callbacks
- 🚀 Streamlit-based frontend for quick UI prototyping
- 🧠 Uses Gaussian-based **density map generation** from scalar crowd labels
- 🔁 Early stopping, LR reduction, and checkpointing during training
- 🗃️ Memory-efficient data loading using `mmap_mode` for large datasets

---

## 🧠 Model Architecture

- **Backbone**: [MobileNetV2](https://arxiv.org/abs/1801.04381) (pretrained on ImageNet, frozen)
- **Decoder**: 5-layer transposed convolution for upsampling
- **Output**: `160x160x1` density map estimating crowd distribution

---

## 📁 Dataset

The dataset is loaded from:
C:/Users/Dell/.cache/kagglehub/datasets/fmena14/crowd-counting/versions/3

yaml
Copy
Edit

- `images.npy`: RGB image array
- `labels.npy`: Crowd count labels (single scalar per image)

> 📌 Density maps are generated **on-the-fly** using 2D Gaussian centered on the image with the magnitude scaled by the crowd count.

---

## 🛠️ Setup Instructions

### 🔗 Install Dependencies

```bash
pip install tensorflow numpy opencv-python matplotlib seaborn streamlit tqdm
You may also need to install:

bash
Copy
Edit
pip install kagglehub
▶️ Run Training (Optional)
bash
Copy
Edit
python app.py
If best_model.h5 already exists, training is skipped.

▶️ Launch the Streamlit App
bash
Copy
Edit
streamlit run app.py
Upload a .jpg, .jpeg, or .png image of a crowd.

The app will:

Predict the density map

Estimate the total count

Display the heatmap overlay

📈 Training Details
Loss: Mean Squared Error (MSE)

Optimizer: Adam (lr = 0.001)

Batch Size: 32

Epochs: 20 (with EarlyStopping)

Callbacks:

ModelCheckpoint

EarlyStopping

ReduceLROnPlateau

Custom memory cleanup callback

🧪 Example Output
Image	Density Map	Estimated Count
🧍🧍🧍🧍🧍	
Estimated Count: 97

📉 Improvements
 Overlay density map on original image

 Support real crowd datasets (UCF-QNRF, ShanghaiTech)

 Add webcam capture support for real-time deployment

 Convert to TensorFlow Lite for edge deployment

🧾 License
This project is open-source under the MIT License.

🙌 Acknowledgments
Dataset from Kaggle: fmena14/crowd-counting

Architecture inspired by:

MCNN

CSRNet

MobileNet + Transposed CNN designs

UI built using Streamlit

🤝 Contributing
Have ideas to improve density map generation, real-time inference, or the model architecture? Contributions are welcome — just open a pull request!
