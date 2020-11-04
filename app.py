import os

from flask import Flask, render_template, request, redirect
from inference import get_prediction

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    error = ""

    if request.method == 'POST':
        img_urls = [request.form['url']]
    else:
        img_urls = ["https://upload.wikimedia.org/wikipedia/en/b/b0/Les-miserables-movie-poster1.jpg"]

    try:
        result = get_prediction(urls=img_urls)
    except:
        img_urls = ["https://upload.wikimedia.org/wikipedia/en/b/b0/Les-miserables-movie-poster1.jpg"]
        result = get_prediction(urls=img_urls)
        error="Invalid URL format"
    return render_template('result.html', img_url=img_urls[0],
                                  class_name=result, error=error)


if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
