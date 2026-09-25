from sklearn.datasets import fetch_california_housing
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor

# Load dataset directly into a Pandas DataFrame
data = fetch_california_housing(as_frame=True)
df = data.frame

# histograms
df.hist()
plt.show()

# create the geographic scatter plot
plt.figure(figsize=(10,7))

scatter = plt.scatter(
    x = df["Longitude"],
    y = df["Latitude"],
    c = df["MedHouseVal"],
    cmap="jet", # color spectrum, blue => cheap, red => expensive
    alpha=0.4,
    s= df["Population"] / 100
)

# add labels and color bar


plt.colorbar(scatter,label = "Median House Value ($100K)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Geographic Distribution of California Housing Prices")
plt.grid(True)

plt.show()

#Split-out validation
array = df.values
X = array[:,0:8]
y = array[:,8]

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size = 0.20,
    random_state = 42
)

# train the model
model = LinearRegression()
model.fit(X_train, y_train)
# 4. Predict on test set
y_pred = model.predict(X_test)

# 5. Evaluate metrics
# Note: root_mean_squared_error is standard in modern scikit-learn

r2 = r2_score(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

print(f"--- Baseline Performance ---")
print(f"R²:   {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# Feature Engineering

df["rooms-per-household"] = df["AveRooms"]
df["bedrooms-per-household"] = df["AveBedrms"] / df["AveRooms"]
df["population-per-household"] = df["AveOccup"]

# Note: If working with raw totals ('TotalRooms', 'TotalBedrooms', 'Households', 'Population'):
# df['rooms_per_household'] = df['total_rooms'] / df['households']
# df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
# df['population_per_household'] = df['population'] / df['households']

# Seperate Features and Target

X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

# Test Train split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Fit Model
model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test,y_pred)
rmse = root_mean_squared_error(y_test,y_pred)

print(f"--- Performance After Feature Engineering ---")
print(f"R²:   {r2:.4f}")
print(f"RMSE: {rmse:.4f}")


# Implementing HistGradientBoostingRegressor 

df["bedrooms-per-household"] = df["AveBedrms"] / df["AveRooms"]

# Seperate Features and Target

X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]


# Test Train split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Fit Model
model = HistGradientBoostingRegressor()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test,y_pred)
rmse = root_mean_squared_error(y_test,y_pred)

print(f"--- Gradient Boosting Performance  ---")
print(f"R²:   {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# 6. Plot Predicted vs. Actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # Perfect fit reference line

plt.xlabel('Actual Median House Value ($100k)')
plt.ylabel('Predicted Median House Value ($100k)')
plt.title('HistGradientBoosting: Predicted vs Actual')
plt.grid(True)
plt.show()
