import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

# Load dataset
df = pd.read_excel("../Dataset/flood dataset.xlsx")

print("Dataset Loaded Successfully")

# Separate features and target
X = df.drop("flood", axis=1)
y = df["flood"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

print("Model Trained Successfully")

# Save model and scaler
joblib.dump(model, "floods.save")
joblib.dump(scaler, "transform.save")

print("Model and Scaler Saved Successfully!")
