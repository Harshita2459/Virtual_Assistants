---

# 🧠 Bias Detection in Virtual Assistants using Machine Learning

This project addresses the critical issue of **bias in text data**—especially within AI systems like virtual assistants and chatbots. The focus is on **detecting, analyzing, and mitigating social biases** using real-world datasets and fairness-enhancing algorithms.

## 📁 Files Included

- `CODE_FILE.py`: Complete Python source code implementing bias detection, mitigation, and fairness evaluation.
- `crows_pairs_anonymized.csv`: Dataset containing text samples annotated for bias type and stereotypicality.
- `prompts.csv`: Supplementary prompts used for bias detection evaluation.
- `VIRTUAL_ASSISTANTS_REPORT.pdf`: Detailed project report describing problem statement, methods, and results.

---

## 🔍 Problem Statement

Bias in AI responses can reinforce harmful stereotypes, especially in sensitive areas like **gender**, **race**, and **religion**. This project aims to:

- **Detect** and quantify bias in NLP datasets.
- **Mitigate** bias using reweighing and adversarial debiasing.
- **Evaluate** model fairness through standard metrics.
- **Deploy** a real-time bias detection module using transformer models.

---

## 🚀 Project Features

### ✅ Data Preprocessing
- Cleans and loads datasets (`crows_pairs_anonymized.csv`, `prompts.csv`)
- Drops unnecessary columns and handles missing values

### 📊 Visualization
- Visualizes **bias type distributions** and **stereotypical vs. anti-stereotypical** text proportions

### ⚖️ Bias Mitigation
- Uses [AIF360 Toolkit](https://aif360.mybluemix.net/) for:
  - **Reweighing**
  - **Adversarial Debiasing**
- Calculates **Disparate Impact** and **Equalized Odds Difference**

### 📈 Model Training
- Trains a logistic regression model on resampled (SMOTE) and debiased data
- Evaluates accuracy alongside fairness metrics

### 🤖 Real-Time Bias Detection
- Leverages HuggingFace’s `unitary/unbiased-toxic-roberta` model
- Classifies user input and flags biased responses

---

## 📦 Python Libraries Used

| Library        | Purpose                                      |
|----------------|----------------------------------------------|
| `pandas`       | Data handling and preprocessing              |
| `seaborn`/`matplotlib` | Data visualization                    |
| `aif360`       | Fairness metrics and bias mitigation         |
| `scikit-learn` | Model training and evaluation                |
| `imbalanced-learn` | SMOTE-based data balancing             |
| `transformers` | Pre-trained NLP models for bias detection    |
| `tensorflow`   | Adversarial debiasing implementation         |

---

## 📊 Sample Results

- **Bias Type Distribution:** Gender and race were the most frequent categories.
- **Bias Mitigation:** Disparate Impact improved from **0.75 ➝ 0.92**.
- **Real-Time Detection:** Successfully flagged toxic/stereotypical responses.

---

## ⚙️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/bias-detection-virtual-assistants.git
   cd bias-detection-virtual-assistants
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the main script:
   ```bash
   python CODE_FILE.py
   ```

---

## 📄 Report

Read the full methodology, architecture, and results in [`VIRTUAL_ASSISTANTS_REPORT.pdf`](./VIRTUAL_ASSISTANTS_REPORT.pdf).

---

## 📚 References

- [AIF360 by IBM](https://aif360.mybluemix.net/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- Bird et al., “Fairness in Machine Learning: A Survey”, *arXiv:2010.04053*

---

## 💡 Future Work

- Integrate with chatbot frameworks (e.g., Rasa, Dialogflow)
- Fine-tune transformers on domain-specific bias data
- Expand to multi-lingual bias detection

---

## 🧑‍💻 Author

Harshita Sinha
Undergraduate Student
Kalinga Institute of Industrial Technology
[GitHub](https://github.com/Harshita2459)

---
