from flask import Flask, request, jsonify
import joblib
import numpy as np

# Cargar el modelo previamente entrenado
modelo = joblib.load("models/modelo_entrenado.joblib")

# Inicializar la aplicación Flask
app = Flask(__name__)

@app.route('/predecir', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_data = np.array(data['input']).reshape(1, -1)
        prediccion = modelo.predict(input_data)
        return jsonify({'prediccion': int(prediccion[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


