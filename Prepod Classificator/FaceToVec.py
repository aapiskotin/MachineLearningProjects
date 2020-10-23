import dlib
import numpy as np


class FaceToVec:

    def __init__(self, shape_pred_data, facerec_data):
        self.sp = dlib.shape_predictor(shape_pred_data)
        self.facerec = dlib.face_recognition_model_v1(facerec_data)
        self.detector = dlib.get_frontal_face_detector()
        self.win = dlib.image_window()


    def load_image(self, image_path):
        return dlib.load_rgb_image(image_path)


    def get_vecs(self, loaded_image):
        face_vec = self.facerec.compute_face_descriptor(loaded_image, self.get_shape(loaded_image))

        return face_vec


    def show_image(self, image, shape=None, set_overlay=False):
        self.win.clear_overlay()
        self.win.set_image(image)

        if shape and set_overlay:
            self.win.add_overlay(shape)


    def get_shapes(self, loaded_image):
        dets = self.detector(loaded_image)
        shape = self.sp(loaded_image, dets)

        return shapes

    def get_best_vec(self, loaded_image):
        faces, scores, _ = self.detector.run(loaded_image)

        if len(faces) == 0:
            return -1

        best_face = faces[np.argmax(scores)]
        shape = self.sp(loaded_image, best_face)

        return self.facerec.compute_face_descriptor(loaded_image, shape)
