# 📈 Stock Market Prediction Using NLP 🚀

Welcome to the ultimate playground for **stock market forecasting with Artificial Intelligence!**  
This repo showcases how **Natural Language Processing (NLP)**, **Deep Learning**, and **Hidden Markov Models** can supercharge your predictions using real-world news and historical price data.

---

## 🗂️ Table of Contents

- ✨ [Overview](#overview)
- 📁 [Project Structure](#project-structure)
- 🧠 [Explanation of All Notebooks & Methods](#explanation-of-all-notebooks--methods)
- ⚙️ [Setup & Requirements](#setup--requirements)
- ▶️ [Usage](#usage)
- 📊 [Results & Evaluation](#results--evaluation)
- 📜 [License](#license)
- 👤 [Author](#author)

---

## ✨ Overview

Predict stock market direction and price using **NLP-powered sentiment analysis**, **stacked LSTM neural networks**, and **Hidden Markov Models (HMM)**!  
Whether you're a data scientist, finance geek, or AI fan, this project will help you explore:

- 📰 **News Sentiment → Stock Movement**
- ⏳ **Historical Price → Future Forecasts**
- 🧩 **Market Regime Detection → Hidden Markov Models**

---

## 📁 Project Structure

```
.
├── Stock Market Prediction and Forecasting using stacked LSTM .ipynb
├── stock price prediction using Sentiment Analysis using Bag of Words.ipynb
├── hmm.ipynb
├── README.md
```

---

## 🧠 Explanation of All Notebooks & Methods

### 1. 📰 Sentiment Analysis using Bag of Words

**File:** `stock price prediction using Sentiment Analysis using Bag of Words.ipynb`  
**Description:**  
Uses daily news headlines to predict whether the stock market will go UP or DOWN.  
**Method:**  
- Headlines are cleaned and concatenated for each day.
- Bag-of-Words model with bigram features transforms text to vectors.
- A Random Forest Classifier is trained to predict market movement based on sentiment.
- Model performance is evaluated using accuracy, precision, recall, and confusion matrix.

---

### 2. 🔮 Stock Price Forecasting with Stacked LSTM

**File:** `Stock Market Prediction and Forecasting using stacked LSTM .ipynb`  
**Description:**  
Forecasts future stock prices using deep learning on historical price data.  
**Method:**  
- Historical stock prices are normalized and split into train/test sets.
- A stacked LSTM neural network (multiple LSTM layers) is built using TensorFlow/Keras.
- The model learns temporal patterns to predict future prices.
- Predictions are visualized and evaluated using RMSE.

---

### 3. 🧩 Stock Market Regime Modeling with Hidden Markov Model (HMM)

**File:** `hmm.ipynb`  
**Description:**  
Uses Hidden Markov Models to analyze and predict stock market regimes or trends based on sequential price or feature data.  
**Method:**  
- HMM is a probabilistic model for sequences, useful for time series data like stock prices.
- The notebook demonstrates how to set up transition and emission probabilities, and apply the HMM for market state inference and regime prediction.
- This approach can identify hidden states (e.g., bull or bear markets) and forecast future movements based on learned patterns.

---

## ⚙️ Setup & Requirements

- 🐍 Python 3.9+
- 📓 Jupyter Notebook
- 📦 Libraries:
  - pandas
  - numpy
  - scikit-learn
  - pandas_datareader
  - matplotlib
  - tensorflow / keras
  - hmmlearn (for HMM method)

Install all dependencies:
```bash
pip install pandas numpy scikit-learn pandas_datareader matplotlib tensorflow hmmlearn
```

---

## ▶️ Usage

1. **Clone the repo:**  
   ```bash
   git clone https://github.com/AdishPadalia26/Stock-Market-Prediction-Using-NLP.git
   ```
2. **Open notebooks in Jupyter:**  
   - `stock price prediction using Sentiment Analysis using Bag of Words.ipynb`
   - `Stock Market Prediction and Forecasting using stacked LSTM .ipynb`
   - `hmm.ipynb`
3. **Run cells sequentially** to explore, experiment, and visualize!

---

## 📊 Results & Evaluation

- **Sentiment Analysis:**  
  Classification metrics for news-driven predictions.

- **Stacked LSTM:**  
  Visual plots of actual vs. predicted prices, RMSE for accuracy.

- **HMM:**  
  Identification of market states, regime transitions, and probability-based predictions.

---

## 📜 License

🔓 For education and research purposes. See repo for details.

---

## 👤 Author

Developed by **Adish Padalia**  
📧 padaliaadish@gmail.com  
🌐 [GitHub: AdishPadalia26](https://github.com/AdishPadalia26)  
🔗 [LinkedIn: adish-padalia-a3768a230](https://www.linkedin.com/in/adish-padalia-a3768a230/)

---

**Contributions, feedback, and stars ⭐ are welcome!**  
Happy predicting! 🚀📈
