# -*- coding: utf-8 -*-

"""
Created on Thu Nov 11 11:47:54 2021

@author: SAMSUNG
"""

from flask import Flask,render_template,request
import pickle
import numpy as np
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict',methods=['POST'])
def predict():
   
   input_x = [float(x) for x in request.form.values()]
   
   x_values = [np.array(input_x)]
   output = cat_b.predict(x_values)
   output=output.item()
 
   if output == 1:
       output = "Less than or Equal to 50k"
   else :
       output = "Greater Than 50k"
       
   return render_template('result.html',prediction_text="The salary of the employee is {} ".format(output))
if __name__=='__main__':
    app.run(port=5000)





