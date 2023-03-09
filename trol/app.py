import pandas as pd
from flask import Flask, render_template, request
from pandas import read_csv
from flask import Flask, render_template, url_for
from decisionHashtag import decisionHashtag
from decisionHesap import decisionHesap
from multinomialHashtag import multinomialHashtag
from multinomialHesap import multinomialHesap
from randomForestHashtag import randomForestHashtag
from randomForestHesap import randomForestHesap

app = Flask(__name__, static_folder="static")


@app.route('/')
def index():
    return render_template('/searchpanel.html')


@app.route('/searchpanel', methods=['POST', 'GET'])
def mak_logic():
    keyword = request.form.get('keyword')
    sa = multinomialHesap()
    predm, keyword1, follower, following, url, name, img, bg, = sa.DownloadData(keyword)
    return render_template('result.html', keyword=keyword1, pred=predm, follower=follower, following=following,
                           url=url,
                           name=name, img=img, bg=bg)


@app.route('/')
@app.route('/hashtag', methods=['POST', 'GET'])
def hashtagy():
    words = request.form.get('words')
    say = multinomialHashtag()
    name = say.scrape(words)
    df = pd.read_csv('data/sonsonuc.csv')

    result = df[["username", "text", "followers", "following", "troldurum"]]
    result2 = result.rename(columns={"username": "Kullanıcı Adı", "text": "Tweet", "followers": "Takipçi Sayısı",
                                     "following": "Takip Edilen Kİşi Sayısı", "troldurum": "Trol Durumu"})

    return render_template('hashtagresult.html', tables=[result2.to_html()],
                           titles=['İlgili Konu Hakkında Konuşan Hesaplar'], name=name)


@app.errorhandler(Exception)
def page_not_found(e):
    print(e)
    error = "Something went wrong, we are checking it! Click here to close "
    return render_template("searchpanel.html", error=error)


if __name__ == '__main__':
    app.run(debug=True)
