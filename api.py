# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 13:17:54 2018

@author: Maddy
"""

from flask import Flask, jsonify
app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
     json_ = request.json
     query_df = pd.DataFrame(json_)
     query = pd.get_dummies(query_df)
     prediction = lr.predict(query)
     return jsonify({'prediction': list(prediction)})
 
 @app.route('/predict', methods=['POST']) # Your API endpoint URL would consist /predict
def predict():
    if lr:
        try:
            json_ = request.json
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(lr.predict(query))

            return jsonify({'prediction': prediction})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')
    


if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line argument
    except:
        port = 12345 # If you don't provide any port then the port will be set to 12345
    lr = joblib.load(model_file_name) # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load(model_columns_file_name) # Load "model_columns.pkl"
    print ('Model columns loaded')
    app.run(port=port, debug=True)
