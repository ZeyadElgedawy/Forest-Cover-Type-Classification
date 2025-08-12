import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import itertools
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("../data/raw/covtype.csv")

print("\nData Frame Head:")
print(df.head())
print("\nData Frame info:")
print(df.info())


numerical_df = df.iloc[:,:10]
num_cols = numerical_df.columns
print("\nData Frame numerical features:")
print(numerical_df.columns)

print("\nData Frame Description:")
print(df.describe())


# plt.figure(figsize=(20, 15))

# for i, col in enumerate(numerical_df.columns):
#     plt.subplot(4, 3, i + 1)  
#     sns.histplot(numerical_df[col], kde=True)
#     plt.title(col)

# plt.tight_layout()
# plt.title("Numerical features distribution")
# plots_save_path = r"..\outputs\Numerical features distribution.png"
# plt.savefig(plots_save_path, bbox_inches='tight', dpi=300)

plt.figure()
sns.heatmap(numerical_df.corr() , annot=True , cmap='coolwarm')
plt.title("Correlation Matrix")
plots_save_path = r"..\outputs\Correlation Matrix.png"
plt.savefig(plots_save_path, bbox_inches='tight', dpi=300)

X = df.drop('Cover_Type' , axis = 1)
y = df['Cover_Type']
y = y - 1    # Map target labels to (0 -> 6) instead of (1 -> 7):

X_train , X_test , y_train , y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42
)

print("\nTraining features description (Numerical Columns):")
print(X_train[num_cols].describe())

print("\nTest features description (Numerical Columns):")
print(X_test[num_cols].describe())

print("\nTraining target Shape:")
print(y_train.shape)
print("\nTest target Shape:")
print(y_test.shape)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train[num_cols])
X_test_scaled = scaler.transform(X_test[num_cols])

X_train_scaled_df = pd.DataFrame(
    X_train_scaled,
    columns=num_cols,
    index=X_train.index
)
X_test_scaled_df = pd.DataFrame(
    X_test_scaled,
    columns=num_cols,
    index=X_test.index
)

print("\nTraining features description after scaling (Numerical Columns) :")
print(X_train_scaled_df.describe())

print("\nTest features description after scaling (Numerical Columns) :")
print(X_test_scaled_df.describe())

X_train_final_scaled = pd.concat([X_train_scaled_df , X_train.drop(num_cols , axis = 1)] , axis = 1)
X_test_final_scaled = pd.concat([X_test_scaled_df , X_test.drop(num_cols , axis = 1)] , axis = 1)

print("\nFinal Training set info:")
print(X_train_final_scaled.info())

print("\nFinal Test set info:")
print(X_test_final_scaled.info())


models = {
    'Random_Forest' : RandomForestClassifier(random_state=42),
    'XGBoost' : XGBClassifier(random_state = 42),
}

results = {}
for name , model in models.items():
    model.fit(X_train_final_scaled , y_train)
    y_pred = model.predict(X_test_final_scaled)
    acc = accuracy_score(y_test , y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    results[name] = y_pred

cm = confusion_matrix(y_test, results['Random_Forest'])
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(1, 8), yticklabels=range(1, 8))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix: Random Forest')
plt.show()

cm = confusion_matrix(y_test, results['XGBoost'])
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(1, 8), yticklabels=range(1, 8))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix: Random Forest')
plt.show()

# For Random Forest
# feat_importances = pd.Series(model['Random_forest'].feature_importances_, index=X.columns)
# feat_importances.nlargest(10).plot(kind='barh', title='Top 10 Features: Random Forest')

# For XGBoost
xgb_importances = pd.Series(models['XGBoost'].feature_importances_, index=X.columns)
xgb_importances.nlargest(10).plot(kind='barh', title='Top 10 Features: XGBoost')

