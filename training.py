import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

# Cargar datos
df = pd.read_csv('data/titanic.csv')  # Ajusta la ruta si es necesario
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
df = df.dropna()
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Separar variables
X = df.drop('Survived', axis=1)
y = df['Survived']

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear y entrenar el modelo
model = LogisticRegression()
model.fit(X_train, y_train)

# Guardar el modelo
with open('models/modelo.pkl', 'wb') as file:
    pickle.dump(model, file)

print("✅ Modelo entrenado y guardado correctamente en models/modelo.pkl")
