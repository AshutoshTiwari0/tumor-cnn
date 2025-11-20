# 🧠 CerebroScan – Brain Tumor Classification using CNN

CerebroScan is a Convolutional Neural Network (CNN) based system that classifies MRI brain images into **four categories**:

- **Glioma**
- **Meningioma**
- **No Tumor**
- **Pituitary**

This project includes a trained CNN model, a Streamlit web app, and downloadable model weights for quick deployment.

---

## 📊 Model Performance

| Class        | Precision | Recall | F1-score | Support |
|--------------|-----------|--------|----------|---------|
| Glioma       | 0.94      | 0.94   | 0.94     | 289     |
| Meningioma   | 0.94      | 0.92   | 0.93     | 385     |
| No Tumor     | 0.98      | 0.98   | 0.98     | 407     |
| Pituitary    | 0.97      | 0.99   | 0.98     | 349     |
| **Accuracy** | **0.96**  | —      | —        | 1430    |

The model achieves **96% accuracy**, demonstrating balanced and strong performance.

---

## 🏗️ Model Architecture (Keras)

```python
model = Sequential()
model.add(Conv2D(64, (3,3), activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(4, activation='softmax'))
```
🌐 Streamlit Application

A simple, interactive Streamlit web app is included.

Features:

Upload MRI images

Image preprocessing (resize to 224×224, normalize)

Real-time prediction

Auto-download model weights if missing

```
📁 Project Structure

📦 tumor-cnn
├── app.py               # Streamlit app
├── tumor.weights.h5     # Auto-downloaded weights
├── README.md
├── requirements.txt
└── images/              # Sample images folder
```
🛠️ Installation & Usage
1. Clone the repository
    ```bash
   git clone https://github.com/AshutoshTiwari0/tumor-cnn.git
   cd tumor-cnn
   ```
2.  Install requirements
   ```bash
    pip install -r requirements.txt
  ```
3. Launch the Streamlit app
   ```bash
   streamlit run app.py
   ```

    <h2>📚 Dataset</h2>
    <p><strong>Dataset includes four MRI classes:</strong></p>

    <ul>
        <li>Glioma</li>
        <li>Meningioma</li>
        <li>No Tumor</li>
        <li>Pituitary</li>
    </ul>

    <p><strong>Dataset Source:</strong> Public MRI Dataset / Kaggle</p>

    <hr style="margin: 25px 0;">

    <h2>🔧 Technologies Used</h2>
    <ul>
        <li>Python</li>
        <li>TensorFlow / Keras</li>
        <li>NumPy</li>
        <li>PIL</li>
        <li>Streamlit</li>
        <li>GitHub Releases</li>
    </ul>

    <hr style="margin: 25px 0;">

    <h2>🚀 Future Improvements</h2>
    <ul>
        <li>Add Grad-CAM heatmaps</li>
        <li>Integrate transfer learning (ResNet, EfficientNet)</li>
        <li>Deploy to Streamlit Cloud / HuggingFace</li>
        <li>Improve UI/UX</li>
    </ul>

    <hr style="margin: 25px 0;">

    <h2>👨‍💻 Author</h2>
    <p><strong>Ashutosh Tiwari</strong></p>

