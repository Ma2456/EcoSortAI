# EcoSortAI — Explainable AI-Based Waste Detection, Classification & Recommendation System

## 📌 Overview

**EcoSortAI** is an Explainable Artificial Intelligence (XAI)-based waste management system designed to automatically detect, classify, explain, and recommend appropriate disposal methods for waste items from uploaded images.

The system combines **YOLO-based object detection**, **MobileNetV3-based image classification**, and explainability techniques such as **LIME, SHAP, and Grad-CAM** to provide not only predictions but also understandable explanations for the model's decisions.

The system can identify multiple waste objects in an image and classify them into categories such as:

- ♻️ Plastic
- 📄 Paper & Cardboard
- 🥫 Metal
- 🍃 Organic
- 💻 E-Waste
- 🧪 Hazardous
- 🪟 Glass

Based on the predicted category, EcoSortAI provides a suitable waste-management recommendation such as recycling, composting, or safe/hazardous disposal.

---

## 🎯 Objectives

The main objectives of EcoSortAI are:

- Automatically detect waste objects from images.
- Classify detected waste into predefined categories.
- Provide explainable AI outputs instead of only predictions.
- Generate visual explanations using Grad-CAM.
- Generate feature-based explanations using LIME and SHAP.
- Recommend appropriate waste disposal methods.
- Provide an interactive web interface using Streamlit.
- Present waste statistics and prediction information through a dashboard.

---

## 🔄 System Workflow

```text
                 ┌─────────────────────┐
                 │     Upload Image     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   YOLO Detection     │
                 │ Multiple Objects     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  Crop Detected       │
                 │      Objects         │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ MobileNetV3          │
                 │ Classification       │
                 └──────────┬──────────┘
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
        ┌─────────────────┐    ┌──────────────────┐
        │  XAI Analysis   │    │ Waste Management │
        │ LIME / SHAP /   │    │ Recommendation   │
        │    Grad-CAM     │    └──────────────────┘
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────────┐
        │ Streamlit Dashboard │
        └─────────────────────┘
```

---

## 🧠 AI Architecture

EcoSortAI uses a hybrid two-stage AI pipeline.

### Stage 1 — Object Detection

A **YOLO-based object detection model** is used to identify and localize multiple waste objects within an uploaded image.

The detector provides:

- Bounding boxes
- Object locations
- Detection confidence
- Multiple-object detection

The detected objects are then cropped and passed to the classification stage.

### Stage 2 — Waste Classification

The detected waste crops are classified using **MobileNetV3**.

The current classification categories include:

```text
ewaste
glass
hazardous
metal
organic
paper_cardboard
plastic
```

The classifier generates the predicted waste category and confidence score.

---

## 🔍 Explainable AI

A major feature of EcoSortAI is that it does not treat AI predictions as a black box.

The system uses multiple explainability techniques.

### Grad-CAM

**Grad-CAM (Gradient-weighted Class Activation Mapping)** generates a visual heatmap showing the regions of an image that contributed most to the classification.

This helps users understand:

> "Which part of the image influenced the model's prediction?"

### LIME

**LIME (Local Interpretable Model-agnostic Explanations)** provides local explanations for individual predictions by identifying image regions that contribute positively or negatively to the prediction.

### SHAP

**SHAP (SHapley Additive exPlanations)** is used to analyze the contribution of features toward the model's prediction.

Together, these methods improve the interpretability and transparency of the AI system.

---

## ♻️ Waste Management Recommendation

After classification, EcoSortAI maps the predicted waste category to an appropriate disposal recommendation.

Example:

| Waste Category | Recommendation |
|---|---|
| Plastic | Recycle through appropriate plastic recycling channels |
| Paper/Cardboard | Recycle or reuse |
| Metal | Send to metal recycling |
| Glass | Send to glass recycling |
| Organic | Compost or dispose through organic waste collection |
| E-Waste | Dispose through authorized e-waste collection/recycling |
| Hazardous | Use appropriate hazardous-waste disposal facilities |

The recommendation system is designed to help users understand what should be done with the detected waste.

---

## 📊 Streamlit Dashboard

EcoSortAI provides an interactive **Streamlit** interface.

The dashboard can display:

- Uploaded images
- Detected objects
- Bounding boxes
- Classification results
- Confidence scores
- Grad-CAM visualizations
- LIME explanations
- SHAP explanations
- Waste-disposal recommendations
- Waste-category statistics
- Prediction information

---

## 🛠️ Technologies Used

### Programming Language

- Python

### Computer Vision

- OpenCV
- YOLO
- Torchvision

### Deep Learning

- PyTorch
- MobileNetV3

### Explainable AI

- LIME
- SHAP
- Grad-CAM

### Data Processing

- NumPy
- Pandas

### Visualization

- Matplotlib
- Plotly

### Web Application

- Streamlit

### Development Environment

- Google Colab
- VS Code
- Git
- GitHub

---

## 📁 Project Structure

```text
EcoSortAI/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   ├── YOLO/
│   └── CNN/
│
├── utils/
│
├── assets/
│
├── data/
│
└── notebooks/
```

> The exact structure may vary depending on the deployment version of the project.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Ma2456/EcoSortAI.git
```

### 2. Navigate to the Project

```bash
cd EcoSortAI
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Environment

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📥 Input

The system accepts an image containing one or more waste objects.

Example:

```text
Input Image
     ↓
YOLO detects objects
     ↓
Objects are cropped
     ↓
MobileNetV3 classifies each crop
```

---

## 📤 Output

For each detected object, the system can provide:

```text
Detected Object
      ↓
Waste Category
      ↓
Confidence Score
      ↓
Explainability Visualization
      ↓
Disposal Recommendation
```

---

## 📈 Model Performance

The classification model achieved approximately **91.5% validation accuracy** during evaluation.

The performance evaluation includes model-level assessment of the classification pipeline and can be extended with additional metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## 🔬 Explainability Pipeline

EcoSortAI follows the following explainability workflow:

```text
              Model Prediction
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       Grad-CAM     LIME       SHAP
          │          │          │
          └──────────┼──────────┘
                     ▼
             Human-Understandable
                  Explanation
```

This allows users and developers to inspect the reasoning behind AI predictions rather than relying only on the final class label.

---

## 🚀 Future Scope

Potential future improvements include:

- Real-time camera-based waste detection
- Mobile application integration
- Expanded waste-category datasets
- Improved detection and classification accuracy
- Multilingual waste-management recommendations
- Location-aware recycling-center recommendations
- Automated waste collection integration
- More advanced XAI techniques
- Cloud-based model serving
- Continuous model improvement using new waste images

---

## 🌱 Impact

EcoSortAI aims to make waste identification and disposal guidance more accessible by combining computer vision with explainable AI.

The system can support:

- Waste segregation awareness
- Recycling education
- E-waste identification
- Hazardous waste awareness
- Sustainable waste-management practices
- AI transparency and education

---

## 👩‍💻 Project

**EcoSortAI**

**Title:**  
**Explainable AI-Based Waste Detection, Classification, and Recommendation System**

Developed as an academic/final-year project in **Computer Science and Design**.

---

## 📄 License

This project is developed for academic and educational purposes.
