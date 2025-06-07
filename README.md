# 🧠 Handwritten Digit Recognition using Ensemble Learning

## 📌 Overview
This project focuses on building a robust handwritten digit recognition system by combining the predictive strengths of **Convolutional Neural Network (CNN)**, **Support Vector Machine (SVM)**, and **Random Forest (RF)** classifiers. By using ensemble learning, we were able to achieve high accuracy and improve generalization on unseen data.

## 📈 Problem Statement
Handwritten digit recognition is a classical machine learning problem, where the goal is to identify digits (0–9) from images. It has real-world applications in:
* Postal code recognition
* Bank cheque digitization
* Automated form processing
This project aims to build a high-performing model that can recognize digits from grayscale images with high accuracy.

## 🧰 Tech Stack
* **Python**
* **TensorFlow / Keras** (for CNN)
* **scikit-learn** (for SVM and Random Forest)
* **Pandas, NumPy, Matplotlib** (for data analysis and visualization)
* **Jupyter Notebook**

## 📂 Dataset
We used the **[Kaggle Digit Recognizer Dataset](https://www.kaggle.com/competitions/digit-recognizer/data)** based on the famous **MNIST dataset**.
* Each image is 28x28 pixels (grayscale)
* Training data: `train.csv` — includes 785 columns (1 label + 784 pixel values)
* Testing data: `test.csv` — includes 784 pixel columns

## 📌 Project Structure

digit-recognition-ensemble/
│
├── train.csv                 # Training dataset
├── test.csv                  # Test dataset
├── digit_recognition.ipynb   # Jupyter Notebook with all code
├── README.md                 # Project documentation

## ⚙️ Workflow
1. **Data Preprocessing**

   * Normalization of pixel values (0–255 → 0–1)
   * Splitting data into training and validation sets

2. **Model Building**

   * **CNN Model**: Designed with Conv2D, MaxPooling, Dense layers
   * **SVM Model**: Trained on flattened pixel data
   * **Random Forest Model**: Trained with default parameters on the same data

3. **Ensemble Strategy**

   * Predict with each model
   * Combine predictions using **majority voting**
   * Evaluate ensemble performance on validation data

4. **Evaluation**

   * CNN Accuracy
   * SVM Accuracy
   * RF Accuracy
   * **Final Ensemble Accuracy: 97.95%**

## 📊 Results

* CNN Accuracy: \~97%
* SVM Accuracy: \~96%
* RF Accuracy: \~95%
* **Ensemble Accuracy**: ✅ **97.95%**

## ✅ Conclusion
By combining the capabilities of deep learning and traditional machine learning models, the ensemble system significantly boosts overall accuracy. This hybrid approach is ideal for handling real-world challenges where different models have complementary strengths.

## 💡 Future Improvements

* Try other ensemble techniques like weighted voting or stacking
* Hyperparameter tuning using grid search
* Use data augmentation to boost CNN performance
👨‍💻 Owner
This project is created and maintained by Sanchit Singh, as part of a practical exploration of machine learning and deep learning techniques for image classification.

🚀 Final Note
This project showcases the power of combining multiple algorithms to solve real-world problems effectively. With an ensemble accuracy of 97.95%, it stands as a strong example of hybrid learning in action. Feel free to explore, contribute, or build upon it!
