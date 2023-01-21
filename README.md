# Tweet-Classifier


大学2年Aセメスターのプログラミング基礎の授業で最終提出物の自由製作課題として作ったものです。

I made this as my final submission for a free production project in my second year Autumn semester programming basics class at university.

## 概要/Overview
ツイート内容を入力するとそのツイートが、バズる・炎上する・その他 のどれになるかを
予測するサービスです。

The service predicts whether a tweet will be a "buzz", "flame", or other.

### ＊デモ動画

デモ動画は[youtube](https://www.youtube.com/watch?v=NEOWWGJ4TqY)にも上がっています。

The demonstration video is also available on [youtube](https://www.youtube.com/watch?v=NEOWWGJ4TqY).


![demo](figures/Original-Service_-Tweet-Classifier.gif)


### ＊プレゼンテーションスライド

授業内の成果発表時に用いた[スライド](###)です。

This [slide](figures/TwitterClassifier_for_upload.pptx) was used during the presentation of the results in class.



## 使い方/Usage
### ＊環境/Dependencies

* Python 3.9.15
* PyTorch 1.9.0
* transformers 4.25.1
* transformers[ja]
* Flask 2.2.2
* Jinja2 3.1.2

### ＊準備/Preparation

このリポジトリをフォークしてからクローンする、もしくはダウンロードしてください。<br>
[ここ](https://drive.google.com/file/d/1nm0raCFzuMWgd8uxV8EJWI0RqzVk2Ui6/view?usp=share_link)からnet_trained.pthというファイルをダウンロードし、クローンまたはダウンロードしたTweet-Classifierのディレクトリの直下にnet_trained.pthを置いてください。<br>
<br>
<br>
Fork this repository and then clone, or download.<br>
Download the file net_trained.pth from [here](https://drive.google.com/file/d/1nm0raCFzuMWgd8uxV8EJWI0RqzVk2Ui6/view?usp=share_link) and place net_trained.pth directly under the clone or downloaded Tweet-Classifier directory.


### ＊遊び方/How to play
ローカルでターミナルを開き、クローンorダウンロードしたTweet-Classifierのディレクトリに移動し、以下のコマンドを打ってください。
```
python main.py
```
するとデバッガ用のサーバーが立ち上がるので(少し時間がかかります)、ブラウザを開いて http://127.0.0.1:5050 を開いてみましょう。<br>
するとホーム画面が出てくると思います。<br>
テキストボックスに判定したい内容を打ち込み、下の「これで判定する」のボタンを押してください。<br>
画面遷移後は、「戻る」のボタンを押すとホーム画面に戻ります。「詳しくみる」のボタンを押すとその判断の根拠(と思われるもの)が表示され、その後「閉じる」のボタンを押すと閉じます。<br>
<br>
<br>
Open terminal in your local , go to that cloned or downloaded Tweet-Classifier directory, and type the following command: 
```
python main.py
```
Then the server for the debugger will be launched (it may take a while). Let's open a browser and go to http://127.0.0.1:5050. <br>
You will then see the home screen of this service.<br>
Type what you want to judge in the text box and press the "これで判定する" button below.<br>
After the screen transition, press the "戻る" button to return to the home screen. Clicking on the "詳しく" button will display (what is thought to be) the basis for the decision, and then clicking on the "閉じる" button will close the section.<br>


## 詳しくは/For more details

これを作った際のことがブログにまとめられています。ぜひ見てやってください、喜びます。

The creation of this is summarized on my blog(Japanese). Please take a look at it, I will be delighted.

## 参考にしたもの/Reference














