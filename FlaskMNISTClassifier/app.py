import os

from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug import secure_filename

from .mnist_predict import predict
from .image_to_npy import img_to_npy


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static/uploads')
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'npy'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.urandom(24)


@app.route('/')
@app.route('/index')
def landing():
    return render_template('index.html')


@app.route('/result/<filename>')
def result(filename):
    rel_path = url_for('static', filename='uploads/'+filename)

    path_to_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img_to_npy(path_to_file)

    filepath_npy = os.path.splitext(path_to_file)[0] + ".npy"
    res, prob = predict(filepath_npy)
    return render_template(
        'result.html',
        image_dir=rel_path,
        res=res,
        prob=prob)


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            img_dir = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_dir)
            return redirect(url_for('result', filename=filename))

    return redirect(url_for('landing'))


# start the server with the 'run()' method
if __name__ == '__main__':
    app.run(debug=True)
