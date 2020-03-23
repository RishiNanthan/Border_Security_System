import cv2
from mtcnn import MTCNN
import threading
from Person import Person
import time
import PySimpleGUI as gui
import io
from PIL import Image


SIZE = 700
gui.theme("DarkAmber")


class Location:

    def __init__(self, src):
        self.vid = cv2.VideoCapture(src)
        self.img = None
        self.persons = []
        self.camera = 1
        if src == 0:
            self.camera = 0
            threading.Thread(target=self.read_images).start()
        self.detector = MTCNN()
        self.run()

    def read_images(self):
        _ = True
        while _:
            _, fr = self.vid.read()
            self.img = cv2.flip(fr, 1)

    def run(self):
        n = 0
        pre = time.time()
        _ = True
        while _:
            if self.camera:
                _, self.img = self.vid.read()

            n += 1
            if time.time() - pre >= 1:
                print(f"{n} fps")
                n = 0
                pre = time.time()

            faces, img = self.detect_faces()

            self.update_trackers(img)

            for face in faces:
                x, y, w, h = face["box"]

                if face["confidence"] < 0.8 or x <= 0 or y <= 0 or w <= 0 or h <= 0:
                    continue

                x1, y1 = x + w, y + h
                found = False
                for person in self.persons:
                    if person.is_same_face(x, y, x1, y1):
                        person.set_face(x, y, x1, y1, img)
                        found = True
                        break
                if not found:
                    person = Person(x, y, x1, y1, face, img)
                    self.persons += [person]

            img = self.draw_persons(img)

            cv2.imshow('Border Security System', img)
            cv2.setMouseCallback('Border Security System', on_mouse=self.mouse_event_handler)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()
        self.vid.release()
        exit(1)

    def detect_faces(self):
        img = self.img
        return self.detector.detect_faces(img), img

    def draw_persons(self, img):
        for person in self.persons:
            img = cv2.rectangle(img, (person.face_x, person.face_y), (person.face_x1, person.face_y1), (0, 200, 0), 1)
            img = cv2.putText(img, person.name, (person.face_x, person.face_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                              (0, 255, 0), 1)
        return img

    def update_trackers(self, img):
        persons = []
        for person in self.persons:
            if person.update_tracker(img):
                persons += [person]
        self.persons = persons

    def mouse_event_handler(self, type, x, y, flag, param):
        if type == cv2.EVENT_LBUTTONUP:
            for person in self.persons:
                if person.face_x < x < person.face_x1 and person.face_y < y < person.face_y1:

                    name = ""

                    if person.name != "Recognising...":
                        face = cv2.imread(f'faces\\{person.name}.jpg')
                        name = person.name
                    else:
                        face = person.get_face()
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (300, 300))
                    image = Image.fromarray(face)
                    bio = io.BytesIO()
                    image.save(bio, format='PNG')
                    face = bio.getvalue()

                    layout = [
                        [gui.T(" ", size=(5, 1)), gui.Image(data=face)],
                        [gui.Text("Name :", size=(10, 1)), gui.InputText(name)],
                        [gui.Text("ID   :", size=(10, 1)), gui.InputText()],
                        [gui.Cancel(), gui.Ok()]
                    ]

                    win = gui.Window("Person", no_titlebar=True, alpha_channel=0.98, resizable=True)
                    win.Layout(layout)
                    print(win.Read())
                    win.Close()


if __name__ == '__main__':
    try:
        loc = Location(0)
        # loc = Location(0)
        loc.run()
    except Exception as e:
        print(e)
        exit(0)
