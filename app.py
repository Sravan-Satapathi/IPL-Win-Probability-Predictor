from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)

# Loading the model
model = pickle.load(open('models/ipl_prob', 'rb'))
data = pd.read_csv('cleaned_data.csv')

@app.route('/', methods=['GET'])
def home():
    teams = sorted(data.batting_team.unique())
    return render_template('index.html', teams=teams)

@app.route('/predict', methods=['POST'])
def predict():
    batting_team = request.form['batting_team']
    bowling_team = request.form['bowling_team']
    city = request.form['city']
    target_runs = int(request.form['target'])
    curr_score = int(request.form['current_score'])
    runs_left = target_runs - curr_score
    overs_bowled = float(request.form['overs_bowled'])
    overs_balls = overs_bowled - int(overs_bowled)
    balls = (6 * int(overs_bowled)) + (overs_balls * 10)
    balls_left = 120 - balls
    wickets_left = 10 - int(request.form['wickets_lost'])
    nrr = (curr_score * 6) / balls
    rrr = (runs_left * 6) / balls_left
    
    data_input = [(city, batting_team, bowling_team, target_runs, runs_left, balls_left, wickets_left, nrr, rrr)]
    df = pd.DataFrame(data_input, columns=['city', 'batting_team', 'bowling_team', 'target_runs', 'runs_left', 'balls_left', 'wickets_left', 'nrr', 'rrr'])
    
    prediction = model.predict_proba(df)
    batting_prob = round(prediction[0][1] * 100, 2)
    bowling_prob = round(100 - batting_prob, 2)
    
    generate_chart(batting_prob, bowling_prob)

    html = render_template('result_container.html', batting_prob=batting_prob, bowling_prob=bowling_prob, batting_team=batting_team, bowling_team=bowling_team)
    
    return jsonify({
        'batting_prob': batting_prob,
        'bowling_prob': bowling_prob,
        'batting_team': batting_team,
        'bowling_team': bowling_team,
        'html': html
    })

def generate_chart(batting_prob, bowling_prob):
    percentages = [batting_prob, bowling_prob]
    colors = ['#4285F4', '#DB4437'] 

    fig, ax = plt.subplots(figsize=(2, 0.5))
    ax.barh([''], [100], color='lightgray', edgecolor='none', linewidth=0)
    left = 0

    for percentage, color in zip(percentages, colors):
        ax.barh([''], percentage, left=left, color=color, height=1, edgecolor='none', linewidth=0)
        left += percentage

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig('static/plot.png', bbox_inches='tight')

if __name__ == "__main__":
    app.run(debug=True)
