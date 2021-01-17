import numpy as np
import os
from fr_utils import genrt_face_encodings,match_face,create_face_database,load_known_encodings,append_face_database

import pandas as pd
from flask import Flask
from flask import request
from flask import jsonify
image_folder = 'images_recog'


app = Flask(__name__)


# Api used for Employee Identification
@app.route('/recognize_face',methods=['GET','POST'])
def recognize_face():
    result={}
    facility_code = request.args.get('facility')
    file_p = request.files['input_image']
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    f=os.path.join(image_folder,file_p.filename)
    file_p.save(f)
    flag,status,unknown_encoding = genrt_face_encodings(f)

    # Debug messages
    # print('Printing known encodings from recognize face')
    # print(known_codings)
    # print('printing unknown encodings')
    # print(unknown_encoding)

    face_df,known_codings = load_known_encodings(facility_code+'_face_db.pkl')
    print("face df")
    print(face_df)
    if unknown_encoding == "Null":
        result['Message']='No Face Found'
    else:
        if match_face(known_codings,unknown_encoding):
            print('Face Match Found')
            result['Face_Matched']='Yes'
        else:
            print('Face Match Not Found')
            result['Face_Matched']='No'
    return jsonify(result)


# Api used for Employee Enrollment
@app.route('/create_face_db',methods = ['GET','POST'])
def create_face_db():
    result = {}
    facility_code = request.args.get('facility')
    print('facility_code is '+facility_code)
    for file in os.listdir():
        print('file name is '+file)
        if file.endswith("test.csv") and file.startswith(facility_code):
            completion_code,completion_message = create_face_database(file)
            print(completion_code)
            print(completion_message)
            result['completion_code'] = completion_code
    return jsonify(result)


# Api used for updating the pickle database file
@app.route('/update_face_db',methods = ['GET','POST'])
def update_face_db():
    result = {}
    facility_code = request.args.get('facility')
    print('facility_code is '+facility_code)
    for file in os.listdir():
        print('file name is '+file)
        if file.endswith("_update.csv") and file.startswith(facility_code):
            completion_code,completion_message = append_face_database(file)
            print(completion_code)
            print(completion_message)
            result['completion_code'] = completion_code
    return jsonify(result)


if __name__ == '__main__':
    print('I am in main and all set to serve you master')
    app.run(port=5005,debug=False)
