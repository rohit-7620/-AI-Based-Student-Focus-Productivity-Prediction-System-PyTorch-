from flask import Flask, render_template, request
from model import predict_focus

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ""
    if request.method == 'POST':
        screen = float(request.form['screen'])
        study = float(request.form['study'])
        sleep = float(request.form['sleep'])
        social = float(request.form['social'])
        breaks = float(request.form['breaks'])

        result = predict_focus([[screen, study, sleep, social, breaks]])

        mapping = {0: "Low Focus", 1: "Medium Focus", 2: "High Focus"}
        prediction = mapping[result]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
