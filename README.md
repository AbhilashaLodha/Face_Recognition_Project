# Face_Recognition_Project
The project enrols and identifies employees using face recognition techniques. The main libraries used are dlib, face_recognition, opencv.  

### Aim
The project is mainly built with the aim of safe employee check-ins amid Covid-19 crisis. Technologies like Fingerprint scanners require that many people touch the same surface, which could potentially spread infection if someone with Covid-19 were to use an unclean scanner. Facial recognition technology is on the other hand, an excellent method for identifying individuals without the risk of close contact amid coronavirus concerns. 
It would be suitable at government, customs, airports, railway stations, enterprises, schools, communities, surveillance, and other crowded public places.

### Code files
The project consists of 2 python files:
- fr_api.py : contains 3 apis for recognising, creating and updating the employee database respectively
- fr_utils.py : contains necessary python function definitions used for the mentioned apis

### Dataset used
"Labelled Faces in the Wild (LFW) Dataset" from Kaggle is being used to build this project and the data is available at https://www.kaggle.com/jessicali9530/lfw-dataset?select=lfw-deepfunneled

### Implemetation
The main idea behind the project is: 
1. Employees' Enrollment - For the starters, employees would get enrolled by creating a pickle database file from employee's facial images. The pickle file would be made from a csv file (provided initially and containing employees' enterprise ids, names, the office facility name, floor number at which they sit and their face image path). The pickle file would store the facial encodings of the employees' faces. A report also gets generated enlisting the status whether the employee's face got added into the database or not. In case more than one face is found, it can become a security breach and thus the report states that particular employee's face wasn't added to the database since more than one face was found.

2. Employees' Identification - For check-in purposes, the employee's face would be matched with the facial encodings present in the database. If a match is found, the employee, thus, gets recognised. The generated report says that the particular employee's face got matched.

3. Updating Employee's Enrollment Database file - The updation of the pickle database file happens according to the following flowchart:
![Screen Shot 2021-01-18 at 1 47 44 AM](https://user-images.githubusercontent.com/77407100/104854888-46238500-592f-11eb-92ec-485f8e295a13.png)

