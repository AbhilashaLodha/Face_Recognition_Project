# This file contains all the helper functions to be used by the api
# genrt_face_embeddings  -- accepts a single image and generate embedding vector for the image
import numpy as np
import face_recognition
import pandas as pd
import csv
import dlib
import cv2
import os

model = "mmod_human_face_detector.dat"
cnn_face_detector = dlib.cnn_face_detection_model_v1(model)


# This function generates and return face encodings which are 128 long
def genrt_face_encodings(image_path):
    image_loaded = face_recognition.load_image_file(image_path)
    # print("image_loaded",image_loaded)

    face_locations = face_recognition.face_locations(image_loaded)
    print("face_locations",face_locations)
    print("length of list is ",len(face_locations))

    if len(face_locations) == 1:
        flag = True
        face_encodings = face_recognition.face_encodings(image_loaded)[0]
        # print("face_encodings",face_encodings)
        encoded_list = face_encodings
        status = "Face Added into DB"
        return flag,status,encoded_list

    elif len(face_locations) > 1:
        flag = False
        conf_list = []
        img = dlib.load_rgb_image(image_path)
        dets = cnn_face_detector(img, 1)
        print("Number of faces detected: {}".format(len(dets)))
        for i, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(
                i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))
            all_items_list = [d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence]
            conf_list.append(all_items_list)

        conf_list = sorted(conf_list, key = lambda x: x[4],reverse=True)
        conf_list = conf_list[0]
        print("conf_list",conf_list)
        left = conf_list[0]
        top = conf_list[1]
        right = conf_list[2]
        bottom = conf_list[3]
        conf_tuple = (top,right,bottom,left)   # (top, right, bottom, left) order
        face_locations = [conf_tuple]
        face_encodings = face_recognition.face_encodings(image_loaded, known_face_locations=face_locations)[0]
        # print("face_encodings",face_encodings)
        encoded_list = face_encodings
        status = "Not Added into DB; More than 1 face found"
        return flag,status,encoded_list

    else:
        flag = False
        status = "Not Added into DB; No Face Found"
        return flag,status,"Null"


# this function compare an image with a list of images and predict if a match was found in the list
# known_faces = list of encoding of known faces; a list must be passed for this argument
# unknown_face = facial encodings not present in the pickle database file
# we should also add a validation to make sure that there should not be more than one person in the image being passed in order to avoid the security breach where
# accidental scanning of an eligible person lead to the passed entry of a wrong person

def match_face(known_faces, unknown_face):
    results = face_recognition.compare_faces(known_faces,unknown_face,tolerance=0.4)
    # Debug messages
    print('Print Results from match_face')
    print(results)

    # using naive method
    res_list = []
    for i in range(0, len(results)) :
        if results[i] == True :
            res_list.append(i)
    print("all indexes ",res_list)
    facility_code = "ABC1"

    face_df = pd.read_pickle(facility_code+'_face_db.pkl')
    print("face df")
    print(face_df)
    abc = []
    for index,i in enumerate(res_list) :
        print(face_df["image_path"][i])
        img = cv2.imread(face_df["image_path"][i], 1)
        path = '/Users/abhilashalodha/Downloads/Face_Recog/many_match_images/3'
        cv2.imwrite(os.path.join(path , "image%s.jpg" % index), img)
        cv2.waitKey(0)
        abc.append(face_df["face_encodings"][i])
        print("abc", abc)

    face_distances = face_recognition.face_distance(abc, face_df["face_encodings"][2])
    print("face_distances",face_distances)

    for i, face_distance in enumerate(face_distances):
        print("The test image has a distance of {:.2} from known image #{}".format(face_distance, i))
        print("- With a normal cutoff of 0.6, would the test image match the known image? {}".format(face_distance < 0.6))
        print("- With a very strict cutoff of 0.5, would the test image match the known image? {}".format(face_distance < 0.5))
        print()

    number_of_matches = np.sum(results)
    print("number_of_matches",number_of_matches)
    if number_of_matches == 1:
        return True
    else:
        return False


# create_face_database accepts either a single image or a list of images, facility_code, floor
#   1 - generate the embedding of the image received in the parameter
#   2 - insert the generated embedding, name or enterprise id, facility_code and floor as a record information in a pkl
# future validation : Need to make sure that file is unique for enterprise ID

def create_face_database(file_path):
    known_encodings = []
    comments = []
    df = pd.read_csv(file_path)
    print("df")
    print(df)
    df2csv = df
    # validation check for facility column
    if df.facility.isnull().any():
        return 'Error','Facility Can not have null values.'

    unique_facilities = df.facility.nunique(dropna=True)
    if unique_facilities == 1:
        facility_name = df['facility'].unique()[0]
    else:
        return 'Error','One file can have data only for one facility.'

    db_file_name = facility_name+'_face_db.pkl'

    # validation check for enterprise_id column
    if df.enterprise_id.isnull().any():
        return 'Error', 'Enterprise ID can not have null values.'

    if not df.enterprise_id.is_unique:
        return 'Error', 'Enterprise ID values can only be unique.'

    for index,row in df.iterrows():
        print("index", index)
        flag,status,image_encoding = genrt_face_encodings(row[4])
        known_encodings.append(image_encoding)
        # print("known_encodings",known_encodings)
        comments.append(status)
        # print("comments",comments)

    df["face_encodings"] = known_encodings
    df2csv["status"] = comments

    df = df[df.status == "Face Added into DB"]
    print("df")
    print(df)

    df = df.drop(['status'],axis=1)
    # df = df[df.face_encodings != "Null"]
    df = df.reset_index(drop=True)

    print("df")
    print(df)

    df.to_pickle(db_file_name)
    print("pickle file")
    print(pd.read_pickle(db_file_name))

    df2csv.columns = map(str.upper, df2csv.columns)
    print("df2csv")
    print(df2csv)
    df2csv.to_csv(facility_name+"_report.csv",na_rep="NA",index=False,columns=['ENTERPRISE_ID','NAME','FACILITY','FLOOR','STATUS'])

    print(facility_name+"_report")
    print(pd.read_csv(facility_name+"_report.csv"))
    return 'Success','Face Database file Created Successfully'


def load_known_encodings(file_path):
    df_encodings = pd.read_pickle(file_path)
    encoding_list = df_encodings['face_encodings'].tolist()
    return df_encodings,encoding_list


def append_face_database(file_path):
    df = pd.read_csv(file_path)

    # validation check for facility column
    if df.facility.isnull().any():
        return 'Error','Facility Can not have null values.'

    unique_facilities = df.facility.nunique(dropna=True)
    if unique_facilities == 1:
        facility_name = df['facility'].unique()[0]
    else:
        return 'Error','One file can have data only for one facility.'

    # validation check for enterprise_id column
    if df.enterprise_id.isnull().any():
        return 'Error','Enterprise ID can not have null values.'

    if not df.enterprise_id.is_unique:
        return 'Error', 'Enterprise Id values can only be unique.'

    # initialisations
    db_file_name = facility_name+'_face_db.pkl'
    face_df, known_codings = load_known_encodings(db_file_name)
    print("face_df")
    print(face_df)

    enterprise_id_list = face_df['enterprise_id'].tolist()
    print("enterprise_id_list",enterprise_id_list)

    df2csv = pd.read_csv(facility_name+"_report.csv")
    enterprise_id_list_df2csv = df2csv['ENTERPRISE_ID'].tolist()
    print("enterprise_id_list_df2csv",enterprise_id_list_df2csv)

    # conditions
    for index,row in df.iterrows():
        print("\n")
        print("index",index)
        # print("row",row)

        flag,status,image_encoding = genrt_face_encodings(row[4])

        if not flag:
            df2csv = df2csv.append({'ENTERPRISE_ID': row[0],'NAME': row[1],'FACILITY': row[2],'FLOOR': row[3],'IMAGE_PATH': row[4],'STATUS': status}, ignore_index=True)
        else:
            if match_face(known_codings,image_encoding):
                print('Face Match Found')

                # code to check if enterprise id exits or not
                if row[0] in enterprise_id_list:
                    print("Enterprise ID exits")
                    pickle_index = enterprise_id_list.index(row[0])
                    print("pickle_index ",pickle_index)
                    df2csv_index = enterprise_id_list_df2csv.index(row[0])
                    print("df2csv_index ",df2csv_index)

                    # code to check if override flag is Yes or No
                    if row[5] == 'Yes':  # should i add more conditions for yes?
                        print("updating...")
                        # extra check
                        if face_df['enterprise_id'][pickle_index]==row[0]:
                            face_df['face_encodings'][pickle_index] = image_encoding
                            df2csv['STATUS'][df2csv_index] = "Face Updated in DB"    #iska alag se index nikaalna pdega

                    else:
                        print("Override Flag is NO for this employee. No details need to be changed for this employee.")

                else:
                    print("Details aren't updated for this employee. Employee details are already there in the database but the Enterprise ID doesn't match. Please check the ENTERPRISE ID once.")
                    df2csv = df2csv.append({'ENTERPRISE_ID': row[0],'NAME': row[1],'FACILITY': row[2],'FLOOR': row[3], 'STATUS': 'Employee details are already there in the database but the Enterprise ID does not match'}, ignore_index=True)

            else:
                print('Face Match Not Found')
                # code to check if enterprise id exits
                if row[0] in enterprise_id_list:
                    print("Enterprise ID exits")
                    pickle_index = enterprise_id_list.index(row[0])
                    print("pickle_index",pickle_index)
                    df2csv_index = enterprise_id_list_df2csv.index(row[0])
                    print("df2csv_index ",df2csv_index)

                    if row[5] == 'Yes':
                        print("updating...")
                        # extra check
                        if face_df['enterprise_id'][pickle_index]==row[0]:
                            face_df['face_encodings'][pickle_index] = image_encoding
                            df2csv['STATUS'][df2csv_index] = "Face Updated in DB"

                    else:
                        print('Override Flag is NO for this employee. No details need to be changed for this employee.')

                else:
                    print("ENTERPRISE ID DO NOT EXIST")
                    # code to append the entire record
                    face_df = face_df.append({'enterprise_id': row[0],'name': row[1],'facility': row[2],'floor': row[3],'image_path': row[4],'face_encodings': image_encoding}, ignore_index=True)
                    df2csv = df2csv.append({'ENTERPRISE_ID': row[0],'NAME': row[1],'FACILITY': row[2],'FLOOR': row[3],'STATUS': 'Newly Added into DB'}, ignore_index=True)

    print("face_df")
    print(face_df)

    print("df2csv")
    print(df2csv)

    df2csv.drop_duplicates(subset="ENTERPRISE_ID",keep="first",inplace=True)
    print("df2csv.columns",df2csv.columns)
    print("df2csv")
    print(df2csv)

    face_df.to_pickle(db_file_name)
    df2csv.to_csv(facility_name+"_report.csv",na_rep="NA",index=False,columns=['ENTERPRISE_ID','NAME','FACILITY','FLOOR','STATUS'])
    return 'Success','Face Database file Updated Successfully'


