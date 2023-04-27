from flask import Flask, render_template, request, url_for, redirect, send_file
import pickle
import os
import pandas as pd


app = Flask(__name__)
app.config['UPLOAD_PATH'] = "files/"
result = ""
f = None


@app.route('/', methods=["POST", "GET"])
def main():
    global result, typef, f
    if request.method == "POST":
        if request.files['file']:
            f = request.files['file']
            f.save(os.path.join(app.config['UPLOAD_PATH'], f.filename))
        elif 'link' in request.form:
            f = request.form['link']

        return redirect(url_for("result"))
    else:
        return render_template('form.html')


@app.route('/result', methods=["POST", "GET"])
def result():
    global f
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    testdf = pd.read_parquet(f).fillna(0)
    ans = model.predict(testdf)
    ans = pd.DataFrame(ans)
    ans = ans.rename(columns={0:'target'})
    res = pd.concat([testdf["wagnum"], testdf["ts_id"], ans["target"]], axis = 1)
    res.to_csv(os.path.join(app.config['UPLOAD_PATH'], r'answer.csv'), index= False )
    return send_file(os.path.join(app.config['UPLOAD_PATH'], 'answer.csv'), as_attachment=True)
    # return result

app.run("0.0.0.0", "81")