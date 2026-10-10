import joblib, pandas as pd
from flask import Flask, request, render_template_string

app = Flask(__name__)
model = joblib.load("salary_model.pkl")
df = pd.read_csv("Salary Data.csv", encoding="utf-8-sig").dropna()
OPTS = {c: sorted(df[c].unique()) for c in ["Gender", "Education Level", "Job Title"]}

PAGE = """<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Salary Predictor</title>
<style>
*{box-sizing:border-box}
body{margin:0;min-height:100vh;font-family:'Segoe UI',system-ui,sans-serif;color:#4a1414;
background:radial-gradient(circle at 20% 10%,#e8836f 0,transparent 40%),radial-gradient(circle at 85% 90%,#c1272d 0,transparent 40%),#fbf3e0;
display:flex;align-items:center;justify-content:center;padding:24px}
.card{width:100%;max-width:760px;background:rgba(255,250,238,.75);backdrop-filter:blur(14px);border:1px solid rgba(193,39,45,.25);
border-radius:22px;padding:34px;box-shadow:0 20px 60px rgba(120,20,20,.25)}
h1{margin:0;font-size:28px}h1 span{background:linear-gradient(90deg,#c1272d,#8f1a1f);-webkit-background-clip:text;color:transparent}
p.sub{margin:6px 0 24px;color:#8a5a50;font-size:14px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:600px){.grid{grid-template-columns:1fr}}
label{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:#8f1a1f}
select,input{width:100%;margin-top:6px;padding:12px 14px;border-radius:10px;border:1px solid #d9b9a8;
background:#fffaf0;color:#4a1414;font-size:15px;outline:none}
select:focus,input:focus{border-color:#c1272d;box-shadow:0 0 0 3px rgba(193,39,45,.2)}
.full{grid-column:1/-1}
button{margin-top:22px;width:100%;padding:14px;border:0;border-radius:12px;font-size:16px;font-weight:600;color:#fff8e7;cursor:pointer;
background:linear-gradient(90deg,#c1272d,#8f1a1f);transition:transform .15s,box-shadow .15s}
button:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(193,39,45,.45)}
.res{margin-top:24px;text-align:center;padding:22px;border-radius:16px;background:#fff1d6;border:1px solid #c1272d;animation:pop .4s}
.res small{color:#8f1a1f;letter-spacing:.08em;text-transform:uppercase;font-size:12px}
.res b{display:block;font-size:44px;margin-top:4px;color:#c1272d}
@keyframes pop{from{opacity:0;transform:scale(.95)}to{opacity:1;transform:none}}
</style></head><body><div class=card>
<h1><span>Salary Predictor</span></h1><p class=sub>Machine-learning estimate based on an employee profile</p>
<form method=post><div class=grid>
<div><label>Age</label><input name=age type=number min=18 max=70 required value="{{v.age}}"></div>
<div><label>Years of Experience</label><input name=exp type=number step=0.5 min=0 max=50 required value="{{v.exp}}"></div>
{% for name,col in [('gender','Gender'),('edu','Education Level')] %}
<div><label>{{col}}</label><select name={{name}}>{% for o in opts[col] %}<option {{'selected' if v[name]==o}}>{{o}}</option>{% endfor %}</select></div>{% endfor %}
<div class=full><label>Job Title</label><select name=job>{% for o in opts['Job Title'] %}<option {{'selected' if v.job==o}}>{{o}}</option>{% endfor %}</select></div>
</div><button>Predict Salary</button></form>
{% if pred %}<div class=res><small>Estimated annual salary</small><b>${{pred}}</b></div>{% endif %}
</div></body></html>"""

@app.route("/", methods=["GET", "POST"])
def index():
    v = {"age": 30, "exp": 5, "gender": "", "edu": "", "job": ""}
    pred = None
    if request.method == "POST":
        v = request.form.to_dict()
        X = pd.DataFrame([{"Age": float(v["age"]), "Gender": v["gender"],
                           "Education Level": v["edu"], "Job Title": v["job"],
                           "Years of Experience": float(v["exp"])}])
        pred = f"{model.predict(X)[0]:,.0f}"
    return render_template_string(PAGE, opts=OPTS, v=v, pred=pred)

if __name__ == "__main__":
    app.run(debug=False)
