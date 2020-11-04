import os

from flask import Flask, render_template, request, redirect
from inference import get_prediction

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # img_bytes = file.read()
        img_urls = ["https://www.indiewire.com/wp-content/uploads/2019/12/beach_bum.jpg?w=510"]
        result = get_prediction(urls=img_urls)
        class_name = format_class_name(class_name)
        return render_template('result.html', class_id=result,
                               class_name=result)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
