# Turbo.az — Car Market Analysis

An end-to-end data science project: 653K+ listings, a complete data pipeline, a Random Forest model (R²=0.956), a Power BI dashboard, and a Streamlit app.

🚀 **Live demo:** [huggingface.co/spaces/shfatime/turboaz-analytics](https://huggingface.co/spaces/shfatime/turboaz-analytics)

---

## Dataset

**Original data (Kaggle):**
The file is not stored in the repo due to its size.
👉 [Kaggle — TurboAz Cars Project](https://www.kaggle.com/datasets/sehriyarmemmedli/turboaz-cars-project)

**Cleaned datasets I prepared (in this repo):**
- `turboaz_datacleaning_visualisation/turboaz_cars_cleaned_iqr.zip` — after column and null cleaning
- `turboaz-streamlitapp/src/data/turboaz_cars_cleaned.zip` — version prepared for the Streamlit app

---

## Project Structure

```
📦 Turbo.az-Analysis
 ├── 📓 datacleaning_clean.ipynb        ← Data cleaning (56→34 columns)
 ├── 📓 datacleaning_outliers.ipynb     ← Outlier analysis and filtering
 ├── 📓 datavisualisation.ipynb         ← EDA and visualization
 ├── 📓 Turboaz_prediction.ipynb        ← Random Forest model (model.pk is generated here)
 ├── 📊 turboaz_datacleaning_visualisation/
 │   └── turboaz_cars_cleaned.zip
 ├── 📊 powerbi/
 │   ├── Turboaz_dashboard.pbix
 │   ├── page1.png
 │   ├── page2.png
 │   └── page3.png
 └── 🐳 turboaz-streamlitapp/
     ├── Dockerfile
     ├── requirements.txt
     └── src/
         ├── streamlit_app.py
         ├── pages/
         │   ├── 1_MarketDashboard.py
         │   ├── 2_Explore.py
         │   └── 3_PricePrediction.py
         ├── models/
         │   ├── train_model.py
         │   ├── price_model.zip
         │   └── model_meta.json
         └── data/
             └── turboaz_cars_cleaned.zip
```

---

## Workflow

### 1. Data Cleaning
- Reduced from 56 columns to 34 (removed duplicates and columns with high missing rates)
- The `Engine` column was split into `horsepower`, `engine_displacement_num`, and `fuel_type`
- The `Condition` column was split into a `Repainted` column using `split()`
- Null values were filled according to a strategy suited to each column's data type
- Date, numeric, and categorical types were corrected
- Removed duplicate rows

### 2. Outlier Analysis
- A combination of IQR, Z-score, and domain-knowledge-based filtering
- Price: upper bound at `quantile(0.995)`
- Mileage: `0–1,000,000 km`
- Horsepower: `70–515 hp` (|Z-score| > 3)
- Engine displacement: `0.1–30 L`
- Final: **619,498 rows**

### 3. Preprocessing & Feature Engineering
- Feature importance was assessed with a baseline Random Forest to identify low-value columns
- Irrelevant/leaky columns (`price`, `currency`, `day`, `hour`, `views`) were dropped
- Outliers were removed using **Isolation Forest** (contamination=0.01) on `kilometrage_num`, `engine_displacement_num`, `at_gucu`, `production_year`
- Data was split into train/test sets (80/20)
- Features were transformed with a `ColumnTransformer` pipeline:
  - **Numerical:** `StandardScaler` (engine displacement, production year, seat count, horsepower)
  - **Log-scaled:** `log1p` + `StandardScaler` (mileage)
  - **One-Hot Encoding:** market type, accident status, owner count, transmission, drivetrain, fuel type
  - **Target Encoding:** city, dealer name, brand, model, color, body type
  - **Binary/passthrough:** barter, loan, salon, vip, featured, condition flags

### 4. Model
- **Algorithm:** Random Forest Regressor
- **R²:** 0.956
- **MAE:** ~2,762 AZN
- **RMSE:** ~6,872 AZN
- **27 features** used (brand, model, year, mileage, horsepower, etc.)

### 5. Streamlit app and Power BI dashboard
- Streamlit app: interactive UI with market dashboard, data exploration, and price prediction pages, served via Docker
- Power BI dashboard: 3-page report visualizing market trends, pricing, and listing statistics

## Setup

```bash
git clone https://github.com/your-username/Turbo.az-Analysis.git
cd Turbo.az-Analysis
pip install -r turboaz-streamlitapp/requirements.txt
```

Download the original dataset from Kaggle and place it in the root folder as `cars.csv`, then run the notebooks in order:

```
1. datacleaning_clean.ipynb
2. datacleaning_outliers.ipynb
3. datavisualisation.ipynb
4. Turboaz_prediction.ipynb   ← model.pk is generated here (large file, not in repo)
```

### Running the Streamlit App Locally

```bash
cd turboaz-streamlitapp
streamlit run src/streamlit_app.py
```

---

## Technologies Used

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Power BI](https://img.shields.io/badge/Power%20BI-F2C811?style=flat&logo=powerbi&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
