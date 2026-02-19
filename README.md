# 🛰️ Satellite Image Classification with Custom CNN

A deep learning project that builds **custom Convolutional Neural Networks (CNNs) from scratch** to classify satellite images into four categories:

- 🌥️ Cloudy  
- 🏜️ Desert  
- 🌱 Green Area  
- 🌊 Water  

This project includes dataset exploration, baseline CNN training, augmentation + regularization improvements, architecture comparison, and misclassification analysis — **without using transfer learning**.

---

## 📁 Project Structure

```text
Image-classifier/
│
├── data/                # Dataset (ignored in repo)
│
├── notebooks/
│   └── cnn.ipynb        # Full training + evaluation notebook
│
├── requirements.txt
├── README.md
└── .gitignore

```


## 📊 Dataset

**Source:** Kaggle (downloaded via `kagglehub`)  
**Classes:** 4 (`cloudy`, `desert`, `green_area`, `water`)  
**Split:** 80% training / 20% validation (shuffle enabled to ensure all classes appear in both splits)

---

## ✅ Project Deliverables Covered

### 📌 Data Exploration & Preprocessing

- Class distribution analysis  
- Sample image visualization  
- Image properties inspection (shape + pixel range)  
- Train/validation split  
- Image resizing to a fixed size  

---

### 📌 Baseline CNN (No Augmentation)

- Custom CNN with **3 convolution blocks**
- Training curves (accuracy & loss)
- Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score

---

### 📌 Improved CNN (With Augmentation)

#### Augmentation using Keras layers:
- `RandomFlip`
- `RandomRotation`
- `RandomZoom`
- `RandomContrast`

#### Regularization:
- `BatchNormalization`
- `Dropout`

---

### 📌 Model Comparison & Analysis

- Comparison table (baseline vs augmented)
- Misclassified image analysis (visual examples)
- Feature map visualization (Conv layer activations)

---

### 📌 Bonus Architecture Experiment

- `GlobalAveragePooling2D()` instead of `Flatten()`
- Compared performance against main models

---

## 🧠 Models Implemented

### 1️⃣ Baseline CNN (No Augmentation)

- 3 Conv blocks: `Conv2D → MaxPooling`
- `Flatten → Dense`
- Softmax output layer for 4 classes

---

### 2️⃣ Augmented CNN (With Regularization)

#### Augmentation layers:
- `RandomFlip` (horizontal)
- `RandomRotation` (0.10)
- `RandomZoom` (0.10)
- `RandomContrast` (0.10)

#### Regularization:
- Batch Normalization after convolution layers
- Dropout before Dense layers

---

### 3️⃣ Bonus CNN (GlobalAveragePooling)

- Replaces `Flatten()` with `GlobalAveragePooling2D()`
- Lower model capacity
- Fewer parameters

---

## 📈 Results Summary

| Model        | Train Acc | Val Acc | Train-Val Gap |
|--------------|----------:|--------:|--------------:|
| Baseline     | 0.9525    | **0.9627** | -0.0102 |
| Augmented    | 0.9456    | 0.8988 | 0.0469 |
| Bonus (GAP)  | 0.9363    | 0.7194 | 0.2169 |

---

## 🔍 Misclassification Analysis (Augmented Model)

- **Total validation samples:** 1126  
- **Correct predictions:** 1057  
- **Incorrect predictions:** 69  
- **Accuracy:** 0.9387  

### Common Misclassification Patterns:

- Cloud patterns resembled reflective water surfaces  
- Sparse greenery looked similar to desert terrain  
- Mixed land/water boundaries created ambiguity  

---

## 🧩 Interpretation

### ✅ Baseline Performed Best

- Highest validation accuracy (**96.27%**)
- No major overfitting detected
- Validation accuracy slightly higher than training due to dataset split randomness

---

### ⚠️ Augmented Model Performed Worse

- Validation accuracy dropped to ~89.9%
- Augmentation may have been too aggressive or unnecessary
- Mild overfitting appears (positive train-val gap)

---

### ❌ Bonus GAP Model Underperformed

- Large train-validation gap (21.7%)
- Model capacity too low compared to Flatten-based models

---

## 🛠️ Technologies Used

- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Scikit-learn  
- kagglehub  

---

## 🚀 How to Run

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

## 2️⃣ Start Jupyter Notebook
```bash
jupyter notebook
```
## 3️⃣ Open the notebook

notebooks/cnn.ipynb

## 4️⃣ Run all cells in order

## 🔮 Future Improvements

Use a stratified train/val/test split

Tune hyperparameters (learning rate, filter sizes, dropout rates)

Try deeper CNN architectures

Add Grad-CAM heatmaps to explain predictions

(Future work) Apply transfer learning with EfficientNet/ResNet
