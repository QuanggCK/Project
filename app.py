from flask import Flask, render_template

from src.analysis import churn_rate_by_category
from src.data_loader import load_raw, summarize
from src.model import run as run_model

app = Flask(__name__)


@app.route("/")
def index():
    df = load_raw()
    info = summarize(df)
    return render_template(
        "index.html",
        n_rows=info["n_rows"],
        n_cols=info["n_cols"],
        duplicated_ids=info["duplicated_ids"],
        columns=info["columns"].to_dict("records"),
        churn=info["churn"].reset_index().to_dict("records"),
        head_cols=list(info["head"].columns),
        head_rows=info["head"].values.tolist(),
    )


@app.route("/factors")
def factors():
    df = load_raw()
    return render_template("factors.html", data=churn_rate_by_category(df))


@app.route("/model")
def model_page():
    return render_template("model.html", r=run_model())


if __name__ == "__main__":
    app.run(debug=True)