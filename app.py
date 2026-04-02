from flask import Flask, request, render_template_string
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

with open('titanic_model.pkl', 'rb') as f:
    model = pickle.load(f)

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Titanic Survival Predictor</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; background-color: #f4f4f4; }
        h1 { text-align: center; color: #333; }
        form { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        label { font-weight: bold; display: block; margin-top: 15px; }
        select, input { width: 100%; padding: 8px; margin-top: 5px; border: 1px solid #ddd; border-radius: 5px; }
        button { width: 100%; padding: 12px; margin-top: 20px; background-color: #333; color: white; border: none; border-radius: 5px; font-size: 16px; cursor: pointer; }
        .result { text-align: center; font-size: 22px; margin-top: 20px; padding: 15px; background: white; border-radius: 10px; font-weight: bold; }
        .survive { color: green; }
        .die { color: red; }
    </style>
</head>
<body>
    <h1>Titanic Survival Predictor</h1>
    {% if prediction %}
    <div class="result {{ css_class }}">
        {{ prediction }}
    </div>
    {% endif %}
    <form action="/predict" method="post">
        <label>Passenger Class:</label>
        <select name="pclass">
            <option value="1">1st Class</option>
            <option value="2">2nd Class</option>
            <option value="3">3rd Class</option>
        </select>
        <label>Sex:</label>
        <select name="sex">
            <option value="0">Female</option>
            <option value="1">Male</option>
        </select>
        <label>Age:</label>
        <input type="number" name="age" placeholder="Enter age" required>
        <label>Fare Paid:</label>
        <input type="number" name="fare" placeholder="Enter fare amount" required>
        <label>Port of Embarkation:</label>
        <select name="embarked">
            <option value="0">Southampton</option>
            <option value="1">Cherbourg</option>
            <option value="2">Queenstown</option>
        </select>
        <label>Family Size (including yourself):</label>
        <input type="number" name="familysize" placeholder="e.g. 1 for alone" required>
        <label>Traveling Alone?</label>
        <select name="isalone">
            <option value="1">Yes</option>
            <option value="0">No</option>
        </select>
        <button type="submit">Predict Survival</button>
    </form>
    {% if prediction %}
    <div class="result {{ css_class }}">
        {{ prediction }}
    </div>
    {% endif %}
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    
    features = pd.DataFrame([[
        int(data['pclass']),
        int(data['sex']),
        float(data['age']),
        float(data['fare']),
        int(data['embarked']),
        int(data['familysize']),
        int(data['isalone'])
    ]], columns=['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'IsAlone'])
    
    prediction = model.predict(features)[0]
    
    if prediction == 1:
        result = "This passenger would SURVIVE"
        css_class = "survive"
    else:
        result = "This passenger would NOT survive"
        css_class = "die"
    
    return render_template_string(HTML, prediction=result, css_class=css_class)

if __name__ == '__main__':
    app.run(debug=True)