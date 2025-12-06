# UFC_Prediction_Model

# 🥊 UFC Fight Outcome Predictor

**CIS 508 Machine Learning in Business - Final Project**  
**Author:** Anthony Nazaruk  
**Arizona State University**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

---

## Project Overview

This project predicts the probability of a UFC fighter winning against an opponent using machine learning. Given two fighters, the model outputs the win probability for each, helping fans, analysts, and enthusiasts understand fight dynamics based on historical data.

**Live Demo:** [UFC Fight Predictor App](https://ufcpredictionmodel-ddb9ny6kvuwxefjgkvaffu.streamlit.app/)

---

## Business Problem

Predicting UFC fight outcomes has significant value for:
- **Sports Betting:** Informing betting odds and strategies
- **Fight Promotion:** Matchmaking decisions for compelling matchups
- **Fighter Training:** Strategic preparation based on opponent analysis
- **Fan Engagement:** Enhanced viewing experience with data-driven insights
#### We aim to present various UFC stakeholders the ability an edge in predicting the outcome of a fight for whatever listed use-case they may have. 
#### Our specific goal is to accurately predict the probability of fighters winning their fights at a level significantly above random guessing to be able to identify edges in sportsbook market lines to make better educated betting and investing decisions. 

---

## Dataset

- **Source:** UFC fight statistics (2019-2025)
- **Records:** 7,398 fights
- **Fighters:** 1,574 unique fighters
- **Features:** 72 features including:
  - Physical attributes (height, weight, reach, age)
  - Fight statistics (win rate, win streak, KO wins, submission wins)
  - Performance metrics (striking accuracy, takedown accuracy, knockdowns)
  - Differential features (height difference, reach advantage, experience gap)

---

## Machine Learning Approach

### Task
Binary Classification with Probability Output  
- **Target:** `fighter_a_won` (1 = Fighter A wins, 0 = Fighter B wins)
- **Output:** P(Fighter A Wins) — probability between 0 and 1

### Initial Models Evaluated
| Model | Runs | Best ROC-AUC |
|-------|------|--------------|
| XGBoost | 20 | **0.719** |
| Logistic Regression | 12 | 0.714 |
| SVM | 9 | 0.708 |
| Neural Network | 6 | 0.700 |
| Random Forest | 8 | 0.696 |
| Decision Tree | 4 | 0.670 |
| KNN | 8 | 0.635 |
| Naive Bayes | 2 | 0.593 |
| Ensemble (Voting/Stacking) | 8 | 0.715 |
* Top performing initial models were subject to an additional 25 combined runs with tuned hyperparameters.
* Best performing tuned models were subject to an additional 6 combinations of ensemble model runs.

### Best Model
**Ensemble_Tuned_WeightedVoting** including the following models:
XGB_l5
XGB_a0.1
LR_C0.15
SVM_linear



### Performance Metrics
| Metric | Score |
|--------|-------|
| ROC-AUC | 0.721 |
| Accuracy | 66.4% |
| F1 Score | 0.664 |
| Brier Score | 0.214 |

---

## Project Structure

```
ufc_prediction_model/
├── app.py                              # Streamlit web application
├── requirements.txt                    # Python dependencies
├── ufc_fights_2019_2025_with_stats.csv # Fighter statistics dataset
├── UFC_Fight_Prediction_Notebook.ipynb # Full ML pipeline notebook
└── README.md                           # Project documentation
```

---

## How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ufc_prediction_model.git
   cd ufc_prediction_model
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

4. **Open in browser**
   Navigate to `http://localhost:8501`

---

## App Features

- **Fighter Selection:** Dropdown menus with 1,574 UFC fighters
- **Win Probability:** Displays probability percentage for each fighter
- **Visual Progress Bars:** Intuitive probability comparison
- **Stats Comparison:** Expandable table showing fighter statistics side-by-side
- **Real-time Predictions:** Instant results powered by XGBoost

*****Disclaimer: When fighter names are swapped between Fighter A and Fighter B, odds may slightly change, we recommend averaging win percentages of both fighters in both Fighter A and B slots to get an accurate percentage to use as a prediction baseline.*****

---

## Technologies Used

- **Python 3.10+**
- **Streamlit** — Web application framework
- **XGBoost** — Gradient boosting classifier
- **scikit-learn** — Preprocessing and model pipeline
- **Pandas & NumPy** — Data manipulation
- **MLflow** — Experiment tracking (Databricks)

---

## MLflow Experiment Tracking

All model runs were logged to Databricks MLflow, including:
- Hyperparameters for each configuration
- Performance metrics (accuracy, precision, recall, F1, ROC-AUC, Brier score)
- Confusion matrices and ROC/PR curves as artifacts
- Model comparison and selection

---

## Future Improvements

- Add real-time fighter stats updates via web scraping
- Incorporate betting odds as additional features
- Implement fight-style matchup analysis
- Add historical head-to-head records
- Deploy model updates with CI/CD pipeline

---

## License

This project is for educational purposes as part of CIS 508 at Arizona State University.

---

## Acknowledgments

- UFC and Kaggle for publicly available fight statistics
- Anthropic Claude for development assistance
- Arizona State University CIS 508 course

---

*Built with ❤️ using XGBoost & Streamlit*
