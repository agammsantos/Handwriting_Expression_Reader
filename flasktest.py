from flask import Flask,render_template,send_from_directory
from PIL import Image
import numpy as np
import pandas as pd
import os, os.path
import joblib as jb

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('canvas.htm')

@app.route('/prediksi',methods=['POST'])
def hasil():
    model=jb.load('modeldigit')
    gambar1=Image.open('./static/test/test.png')
    grey=gambar1.convert('L')
    gambar2=grey.resize((45, 45))
    gambartest=np.array(gambar2.getdata())
    angka=model.predict([gambartest])
    return render_template('hasil.htm',x=angka)

if __name__=='__main__':
    app.run(debug=True)