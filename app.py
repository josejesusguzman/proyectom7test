from flask import Flask
from flask import request
import pandas as pd
import joblib

app = Flask(__name__)

@app.route('/predecir', methods=['POST'])
def predecir() :
    json_ = request.json
    df = pd.DataFrame(json_, index=[0])
    query = pd.get_dummies(df)

    clasificador = joblib.load('classifier.pkl')
    prediccion = clasificador.predict(query)

    if prediccion[0] == True :
        return "El pasajero pudo haber sobrevivido al titanic"
    else :
        return "El pasajero pudo NO haber sobrevivido al titanic"
    
if __name__ == 'main':
    app.run(port=5000, debug=True)
