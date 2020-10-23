from annoy import AnnoyIndex
import pickle as pkl
from FaceToVec import FaceToVec

VEC_SIZE = 128
annoy_metric = 'euclidean'

class KNN:

    def __init__(self, vector_space, map_id_to_fname, shape_pred_data, facerec_data, teachers_path='./teachers/1/'):
        self.vec_space = AnnoyIndex(VEC_SIZE, metric=annoy_metric)
        self.vec_space.load(vector_space)

        self.mapping = pkl.load(open(map_id_to_fname, 'rb'))
        
        self.ftv = FaceToVec(shape_pred_data, facerec_data)
        self.teachers_path = teachers_path


    def get_nn_fname(self, image_path):
        img = self.ftv.load_image(image_path)
        vec = self.ftv.get_best_vec(img)
        if vec == -1:
            return -1
        else:
            nn_idx = self.vec_space.get_nns_by_vector(vec, 1)[0]

        return self.teachers_path + self.mapping[nn_idx]

    def get_nn_from_frame(self, frame):
        vec = self.ftv.get_best_vec(frame)
        if vec == -1:
            return -1
        else:
            nn_idx = self.vec_space.get_nns_by_vector(vec, 1)[0]

        return self.teachers_path + self.mapping[nn_idx]


    def show_image(self, image_path, shape=None, set_overlay=False):
        self.ftv.show_image(self.ftv.load_image(image_path), shape, set_overlay)

