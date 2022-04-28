
# importing the necessary dependencies
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
from datetime import datetime
from Casendra_database.astra import getDBSession, createJSONonAstra
import pickle

app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            features = ['national_inv', 'lead_time', 'sales_1_month', 'pieces_past_due', 'perf_6_month_avg',
            'local_bo_qty', 'deck_risk', 'oe_constraint', 'ppap_risk', 'stop_auto_buy', 'rev_stop']
            cat_col = ['deck_risk', 'oe_constraint', 'ppap_risk', 'stop_auto_buy', 'rev_stop']
            predict_value = []
            casendra_database = [] #storing data to send to casendra
            date_added = datetime.now() #to store the time when data was added
            casendra_database.append(date_added)
            for feature in features:
                value = request.form[feature]
                if feature in cat_col:
                    casendra_database.append(value)
                    if(value=='yes'):
                        predict_value.append(1)
                    else:
                        predict_value.append(0)
                else:
                    predict_value.append(float(value))
                    casendra_database.append(int(value))

            filename = 'PredictModel.pickle'
            loaded_model = pickle.load(open(filename, 'rb')) # loading the model file from the storage
            prediction=loaded_model.predict([predict_value])

            if prediction == 1:
                prediction = "Product went to Back order"
            else:
                prediction = "Product didn't go to Back order"
            
            # exporting the values to casendra database
            casendra_database.append(prediction)
            sessions = getDBSession()
            data_message = createJSONonAstra(sessions, casendra_database)
            if data_message != 'Successful':
                data_message = f"Something went wrong please see the message --> {data_message}"
            else:
                data_message = f"Data written to the casendra database is {data_message}"

            print('prediction is', prediction)
            # showing the prediction results in a UI
            return render_template('results.html',prediction=prediction, data_message=data_message)
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')



if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
	app.run(debug=True) # running the app