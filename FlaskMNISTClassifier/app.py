from flask import Flask, render_template
from flask import request, flash, redirect, url_for
import os
from werkzeug import secure_filename

UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.urandom(24)


@app.route('/')
@app.route('/index')
def landing():
    return render_template('minimal.html')


@app.route('/result/<filename>')
def result(filename):
    return render_template(
        'result.html',
         image_dir=url_for('static', filename="uploads/"+filename), res=5)


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        app.logger.info("post?!")
        # check if the post request has the file part
        if 'file' not in request.files:
            app.logger.info("no file")
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
