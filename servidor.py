from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
modelo = joblib.load('modelo_oximetro.pkl')

@app.route('/api/v1/data', methods=['POST'])
def recibir_datos():
    datos = request.get_json()
    bpm = datos.get('bpm')
    spo2 = datos.get('spo2')

    if bpm is None or spo2 is None:
        return jsonify({'error': 'Datos incompletos'}), 400

    entrada = np.array([[bpm, spo2]])
    prediccion = modelo.predict(entrada)[0]

    if prediccion == 1:
        print(f"🔴 ALERTA: BPM={bpm}, SpO2={spo2}")
    else:
        print(f"🟢 Normal: BPM={bpm}, SpO2={spo2}")

    return jsonify({'estado': 'recibido', 'anormal': int(prediccion)}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)