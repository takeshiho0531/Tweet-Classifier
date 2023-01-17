import os
import random
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import re

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

#前処理
def extract_new_text(new_text):
  new_text=re.sub('\n',"",new_text)
  new_text=re.sub('\t',"",new_text)
  new_text=re.sub('\r',"",new_text)
  new_text=re.sub('\u3000',"",new_text)
  return new_text

def tokenizer_512(input_text):
    """torchtextのtokenizerとして扱えるように、512単語のpytorchでのencodeを定義。ここで[0]を指定し忘れないように"""
    return tokenizer.encode(input_text, max_length=512, return_tensors='pt')[0]

#HTMLを作成する関数
def highlight(word,attn,pred):
    """ Attentionの値が大きいと文字の背景が濃い赤になるhtmlを出力させる関数 """
    if pred ==3:
        html_color= "#%02X%02X%02X" % (
          int(255*(1-attn)),int(255*(1-attn)),255)
    else:
        html_color= "#%02X%02X%02X" % (
          255, int(255*(1-attn)),int(255*(1-attn)))

    return '<span style="background-color:{}">{}</span>'.format(html_color,word)

#HTMLを作成する関数
def mk_html_new(input,preds,attention_weights):
    """ HTMLデータを作成する """

    #indexの結果を表示
    sentence= input #文章のid列
    pred=preds #予測

    if pred==2:
        pred_str="このツイートはバズる可能性があります。ツイートしてみんなの愛を受け止めましょう。"
    elif pred==1:
        pred_str="つまらないツイートですね。"
    else:
        pred_str="このツイートは炎上する可能性があります。今日の晩御飯は鶏の丸焼きですかね。"

    html="判定: {}<br><br?".format(pred_str)

    #Self-Attentionの重みを可視化。
    #Multi-Headが12個なので、12種類のアテンションが存在
    for i in range(12):
        attens=attention_weights[0,i,0,:]
    attens/=attens.max()

    #12種類のAttentionの平均を求める。最大値で規格化
    all_attens=attens*0 #all_attensという変数を作成する
    for i in range(12):
        all_attens+=attention_weights[0,i,0,:]
    all_attens/=all_attens.max()

    html_output=""
    for word,attn in zip(sentence,all_attens):

        #単語が[SEP]の場合は文章が終わりなのでbreak
        if tokenizer.convert_ids_to_tokens([word.numpy().tolist()])[0]=="[SEP]":
            break

        #関数highlightで色をつける、関数tokenizer_bert.convert_ids_to_tokensでIDを単語に戻す
        html_output+=highlight(tokenizer.convert_ids_to_tokens([word.numpy().tolist()])[0],attn,pred)


    html+=html_output
    html+="<br><br>"

    return html

def remove_str_start_end(s, start, end):
    return s[:start] + s[end + 1:]

from transformers import BertModel

# BERTの日本語学習済みパラメータのモデルです
model= BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking', output_attentions=True)

from torch import nn


class BertForTweetClassifier(nn.Module):
    '''BERTモデルにLivedoorニュースの9クラスを判定する部分をつなげたモデル'''

    def __init__(self):
        super(BertForTweetClassifier, self).__init__()

        # BERTモジュール
        self.bert = model  # 日本語学習済みのBERTモデル

        # headにポジネガ予測を追加
        # 入力はBERTの出力特徴量の次元768、出力は9クラス
        self.cls = nn.Linear(in_features=768, out_features=3)

        # 重み初期化処理
        nn.init.normal_(self.cls.weight, std=0.02)
        nn.init.normal_(self.cls.bias, 0)

    def forward(self, input_ids):
        '''
        input_ids： [batch_size, sequence_length]の文章の単語IDの羅列
        '''

        # BERTの基本モデル部分の順伝搬
        # 順伝搬させる
        result = self.bert(input_ids)  # reult は、sequence_output, pooled_output
        attentions = result['attentions']

        # sequence_outputの先頭の単語ベクトルを抜き出す
        vec_0 = result[0]  # 最初の0がsequence_outputを示す
        vec_0 = vec_0[:, 0, :]  # 全バッチ。先頭0番目の単語の全768要素
        vec_0 = vec_0.view(-1, 768)  # sizeを[batch_size, hidden_size]に変換
        output = self.cls(vec_0)  # 全結合層

        return output, attentions

def inference(new_text):
    new_text=extract_new_text(new_text)
    input=tokenizer_512(new_text)
    input2D=input.unsqueeze(0)

    # モデルの設定
    net_trained=torch.load('./net_trained.pth')
    net_trained.eval() 

    #推論
    outputs, attentions = net_trained(input2D)
    _, preds = torch.max(outputs, 1)

    #結果
    html_output=mk_html_new(input,preds,attentions[-1])
    #print(html_output[0])
    #print(html_output)
    
    if len(html_output)>4:
        for i in range(1,len(html_output)-2):
            if html_output[i]=="#" and html_output[i-1]!=":":
                html_output=remove_str_start_end(html_output,i,i)
            return html_output
    else:
        return "判定不可能でした"
