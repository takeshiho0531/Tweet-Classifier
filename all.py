import re

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')


# 前処理
def extract_new_text(new_text):
    new_text = re.sub('\n', "", new_text)
    new_text = re.sub('\t', "", new_text)
    new_text = re.sub('\r', "", new_text)
    new_text = re.sub('\u3000', "", new_text)
    return new_text


def tokenizer_512(input_text):
    """torchtextのtokenizerとして扱えるように、512単語のpytorchでのencodeを定義。ここで[0]を指定し忘れないように"""
    return tokenizer.encode(input_text, max_length=512, return_tensors='pt')[0]


# HTMLを作成する関数
def highlight(word, attn, pred):
    """ Attentionの値が大きいと文字の背景が濃い赤になるhtmlを出力させる関数 """
    if pred == 3:
        html_color = "#%02X%02X%02X" % (
          int(255*(1 - attn)), int(255*(1 - attn)), 255)
    else:
        html_color = "#%02X%02X%02X" % (
          255, int(255*(1 - attn)), int(255*(1 - attn)))

    return '<span style="background-color:{}">{}</span>'.format(html_color, word)


# HTMLを作成する関数
def mk_html_new(input, preds, attention_weights):
    """ HTMLデータを作成する """

    # indexの結果を表示
    sentence = input  # 文章のid列
    pred = preds  # 予測

    if pred == 2:
        pred_str = "このツイートはバズる可能性があります。ツイートしてみんなの愛を受け止めましょう。"
    elif pred == 1:
        pred_str = "つまらないツイートですね。"
    else:
        pred_str = "このツイートは炎上する可能性があります。今日の晩御飯は鶏の丸焼きですかね。"

    html = "判定: {}<br><br?".format(pred_str)

    # Self-Attentionの重みを可視化。
    # Multi-Headが12個なので、12種類のアテンションが存在
    for i in range(12):
        attens = attention_weights[0, i, 0, :]
    attens /= attens.max()

    # 12種類のAttentionの平均を求める。最大値で規格化
    all_attens = attens*0  # all_attensという変数を作成する
    for i in range(12):
        all_attens += attention_weights[0, i, 0, :]
    all_attens /= all_attens.max()

    html_output = ""
    for word, attn in zip(sentence, all_attens):

        # 単語が[SEP]の場合は文章が終わりなのでbreak
        if tokenizer.convert_ids_to_tokens([word.numpy().tolist()])[0] == "[SEP]":
            break

        # 関数highlightで色をつける、関数tokenizer_bert.convert_ids_to_tokensでIDを単語に戻す
        html_output += highlight(tokenizer.convert_ids_to_tokens([word.numpy().tolist()])[0], attn, pred)

    html += html_output
    html += "<br><br>"

    return html
