import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv("Salary Data.csv", encoding="utf-8-sig").dropna(subset=["Salary"])
X, y = df.drop(columns="Salary"), df["Salary"]
cat = ["Gender", "Education Level", "Job Title"]
num = ["Age", "Years of Experience"]
pre = ColumnTransformer([
    ("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                    ("o", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num)])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
models = {"LinearRegression": LinearRegression(),
          "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
          "GradientBoosting": GradientBoostingRegressor(random_state=42)}
best, best_r2 = None, -9
print(f"{'Model':18}{'R2':>8}{'MAE':>10}{'RMSE':>10}")
for n, m in models.items():
    p = Pipeline([("pre", pre), ("m", m)]).fit(Xtr, ytr)
    pr = p.predict(Xte)
    r2 = r2_score(yte, pr)
    print(f"{n:18}{r2:8.3f}{mean_absolute_error(yte, pr):10.0f}{mean_squared_error(yte, pr)**.5:10.0f}")
    if r2 > best_r2: best, best_r2, bn = p, r2, n
print("Best:", bn)
joblib.dump(best, "salary_model.pkl")
new = pd.DataFrame([{"Age": 35, "Gender": "Male", "Education Level": "Master's",
                     "Job Title": "Software Engineer", "Years of Experience": 10}])
print("Predicted salary:", round(best.predict(new)[0]))
