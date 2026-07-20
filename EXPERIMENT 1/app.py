from flask import Flask, render_template, request
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/visualization")
def visualization():
    return render_template("visualization.html")


@app.route("/formula")
def formula():
    return render_template("formula.html")


@app.route("/result", methods=["POST"])
def result():

    algorithm = request.form["algorithm"]
    xvalues = request.form["xvalues"]
    yvalues = request.form["yvalues"]

    x = list(map(float, xvalues.split(",")))
    y = list(map(float, yvalues.split(",")))

    X = np.array(x).reshape(-1, 1)
    Y = np.array(y)

    model = LinearRegression()
    model.fit(X, Y)

    prediction = model.predict(X)

    # ---------- Regression Graph ----------
    plt.figure(figsize=(6, 4))

    plt.scatter(X, Y, color="blue", label="Actual Data")
    plt.plot(X, prediction, color="red", linewidth=2, label="Regression Line")

    plt.title("Linear Regression")
    plt.xlabel("X Values")
    plt.ylabel("Y Values")
    plt.legend()
    plt.grid(True)

    graph_path = os.path.join("static", "regression.png")
    plt.savefig(graph_path)
    plt.close()

    # ---------- Evaluation Metrics ----------
    slope = model.coef_[0]
    intercept = model.intercept_

    mse = mean_squared_error(Y, prediction)
    mae = mean_absolute_error(Y, prediction)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y, prediction)

    # ---------- Metrics Bar Chart ----------
    plt.figure(figsize=(6, 4))

    metrics = ["MSE", "MAE", "RMSE", "R²"]
    values = [mse, mae, rmse, r2]

    plt.bar(metrics, values)

    plt.title("Evaluation Metrics")
    plt.ylabel("Value")

    metrics_path = os.path.join("static", "metrics.png")

    plt.savefig(metrics_path)
    plt.close()

    return render_template(
    "result.html",
    algorithm=algorithm,
    slope=round(slope, 2),
    intercept=round(intercept, 2),
    equation=f"Y = {slope:.2f}X + {intercept:.2f}",
    mse=round(mse, 2),
    mae=round(mae, 2),
    rmse=round(rmse, 2),
    r2=round(r2, 2)
)


if __name__ == "__main__":
    app.run(debug=True)