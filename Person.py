import cv2
import numpy as np
import threading
import os
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from scipy.spatial.distance import cosine
import random


faces = []
faces_dir = "faces"
Recogniser = VGGFace(model="resnet50", input_shape=(224, 224, 3), include_top=False, pooling="avg")


class Person:
    ID = 1

    def __init__(self, x, y, x1, y1, properties, img):
        self.name = "Recognising..."
        self.name_id = f"Person {Person.ID}"
        Person.ID += 1
        self.face_x = x
        self.face_y = y
        self.face_x1 = x1
        self.face_y1 = y1
        self.tracker = cv2.TrackerKCF_create()
        self.tracker.init(img, (x, y, x1-x, y1-y))
        self.properties = properties
        self.img = img
        # threading.Thread(target=self.recognise_face).start()
        self.recognise_face()

    def set_face(self, x, y, x1, y1, img):
        self.img = img
        self.face_x, self.face_y, self.face_x1, self.face_y1 = x, y, x1, y1
        self.tracker.init(img, (x, y, x1-x, y1-y))
        # threading.Thread(target=self.recognise_face).start()
        self.recognise_face()

    def get_face(self):
        return self.img[self.face_y: self.face_y1, self.face_x: self.face_x1]

    def is_same_face(self, x, y, x1, y1):
        mx = (x + x1) // 2
        my = (y + y1) // 2
        if self.face_x < mx < self.face_x1 and self.face_y < my < self.face_y1:
            return True
        return False

    def recognise_face(self):
        global faces
        face = self.img[self.face_y: self.face_y1, self.face_x: self.face_x1]

        face = np.asarray(face, 'float32')
        face = preprocess_input(face)
        face = cv2.resize(face, (224, 224))
        cv2.imwrite(f'check\\img{random.randrange(1, 1000)}.jpg', face)
        encodings = Recogniser.predict([[face]])
        encoding = encodings[0]

        for i in faces:
            comparison = compare_faces(i.face_encoding, encoding)
            if comparison > 0.5:
                self.name = i.name
                break

    def update_tracker(self, img):
        self.img = img
        _, bb = self.tracker.update(img)
        x, y, w, h = (int(i) for i in bb)
        x1 = x + w
        y1 = y + h
        if x == y == x1 == y1 == 0:
            return False
        self.face_x, self.face_y, self.face_x1, self.face_y1 = x, y, x1, y1
        return _


class Face:

    def __init__(self, encoding, name):
        self.face_encoding = encoding
        self.name = name

    def __repr__(self):
        return self.name


def get_known_face_encodings(folder_path):
    global faces
    face_files = os.listdir(folder_path)
    for file in face_files:
        face = cv2.imread(f"{folder_path}\\{file}")

        face = cv2.resize(face, (224, 224))
        face = np.asarray(face, 'float32')
        face = preprocess_input(face)
        encodings = Recogniser.predict([[face]])
        faces += [Face(encodings[0], file[:-4])]

    print(f"No of Faces in database : {len(faces)}")
    print("People :", *faces)


def compare_faces(known_encoding, unknown_encoding):
    """
    :param known_encoding: encoding found using neural net
    :param unknown_encoding: encoding found using neural net
    :return: score of match 1 - complete match; 0 - no match;
    """
    score = cosine(known_encoding, unknown_encoding)
    return 1 - score


get_known_face_encodings(faces_dir)

if __name__ == '__main__':
    pass
