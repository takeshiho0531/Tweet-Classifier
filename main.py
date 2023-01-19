import torch
from flask import Flask, render_template, request
from IPython.display import HTML

from all import extract_new_text, mk_html_new, tokenizer_512
import all
from all import extract_new_text, tokenizer_512, mk_html_new
from model import BertForTweetClassifier

app = Flask(__name__, static_folder="./static")


@app.route("/")
def default():
    return render_template(
        "default.html"
    )


@app.route("/nextpage", methods=["GET"])
def nextpage():
    return render_template("default.html")


def inference(new_text):
    new_text = extract_new_text(new_text)
    input = tokenizer_512(new_text)
    input2D = input.unsqueeze(0)

    # モデルの設定
    net_trained = torch.load('./net_trained.pth')
    net_trained.eval()

    # 推論
    outputs, attentions = net_trained(input2D)
    _, preds = torch.max(outputs, 1)

    # 結果
    html_output = mk_html_new(input, preds, attentions[-1])

    html_output = html_output.replace("##", "")
    html_output = html_output.replace("[CLS]", "")
    html_output = html_output.replace("[UNK]", "")

    return preds, html_output


@app.route("/result", methods=["POST"])
def result():
    new_text = request.form["your_tweet"]
    preds, html_output = inference(new_text)

    if preds == 0:
        return render_template(
            "Flaming.html",
            html_output=HTML(html_output)
        )
    elif preds == 1:
        return render_template(
            "Other.html",
            html_output=HTML(html_output)
        )
    elif preds == 2:
        return render_template(
            "Buzz.html",
            html_output=HTML(html_output)
        )


if __name__ == ('__main__'):
    app.run(debug=True, host='0.0.0.0', port=5050)
