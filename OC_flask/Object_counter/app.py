
from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
from pred_utils import predict
from flask import session ,jsonify,make_response ,request

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

@app.route('/', methods=['GET',"POST"])
@app.route('/home', methods=['GET',"POST"])
def home():
    file1 =request.files['img']
    file2 =request.files['text']
    print(f"######### {file1} #############")

    #file1 = form.file1.data
    #file2 = form.file2.data 
        # First grab the file
    upload_path=os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'])
    file1_path=os.path.join(upload_path ,secure_filename(file1.filename))
    file2_path=os.path.join(upload_path ,secure_filename(file2.filename))
    
    file1.save(file1_path) # Then save the file
    file2.save(file2_path)
    pred=predict(upload_path,file1_path,file2_path)
    return make_response(jsonify({"Count":pred}))

    
#        return render_template('index.html', form=form ,prediction =pred)
#    return render_template('index.html', form=form ,prediction="" )

if __name__ == '__main__':
    app.run(debug=True)
