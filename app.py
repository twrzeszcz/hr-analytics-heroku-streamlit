import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from flask_cors import cross_origin
import pickle
from pycaret.classification import *
from utils import custom_transformer, skewness_remover

app = Flask(__name__)
prep_pipe = pickle.load(open('prep_pipe.pkl', 'rb'))
stacking_model = load_model('stacking_clf')
col_names = ['enrollee_id', 'city', 'city_development_index', 'gender',
       'relevent_experience', 'enrolled_university', 'education_level',
       'major_discipline', 'experience', 'company_size',
       'company_type', 'last_new_job', 'training_hours']

@app.route('/')
@cross_origin()
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
@cross_origin()
def predict():
    if request.method == 'POST':
        output = [x for x in request.form.values()]
        if output[5:-1] in ['NaN', 'nan', 'NAN']:
            output[5:-1] = np.NaN
        if output[3] in ['NaN', 'nan', 'NAN']:
            output[3] = np.NaN
        output[0] = float(output[0])
        output[2] = float(output[2])
        output[-1] = float(output[-1])
        df = pd.DataFrame([output], columns=col_names)
        df_prep = prep_pipe.transform(df)
        prediction = predict_model(stacking_model, data=df_prep)['Label'].values[0]
        return render_template('index.html',
                               prediction_text='Predicted Label: {}'.format(prediction))
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)