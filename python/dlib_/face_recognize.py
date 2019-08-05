from base64 import b64decode
from copy import copy
from io import BufferedReader, BytesIO
from os import path

import dlib
import numpy as np
from PIL import Image
from pkg_resources import resource_filename


def pose_model_location(file_name):
    file_name = resource_filename(__name__, path.join('models', file_name))
    if not path.exists(file_name):
        raise FileNotFoundError('file {} not found'.format(file_name))
    return file_name


face_detector = dlib.get_frontal_face_detector()
cnn_face_detector = dlib.cnn_face_detection_model_v1(pose_model_location('mmod_human_face_detector.dat'))
pose_predictor_5_point = dlib.shape_predictor(pose_model_location('shape_predictor_5_face_landmarks.dat'))
pose_predictor_68_point = dlib.shape_predictor(pose_model_location('shape_predictor_68_face_landmarks.dat'))
face_encoder_v1_1 = dlib.face_recognition_model_v1(pose_model_location('dlib_face_recognition_resnet_model_v1_1.dat'))
face_encoder_v1_2 = dlib.face_recognition_model_v1(pose_model_location('dlib_face_recognition_resnet_model_v1_2.dat'))


def face_distance(face_encodings_, face_to_compare):
    if face_encodings_ is None:
        return np.empty((0,))
    return np.linalg.norm(face_encodings_ - face_to_compare, axis=1)


def face_compare(face_encoding_, face_to_compare):
    if not face_encoding_:
        return 1
    return np.linalg.norm(face_encoding_ - face_to_compare, axis=0)


def _rect_to_bound(rect, shape):
    return dlib.rectangle(max(rect.left(), 0), max(rect.top(), 0), min(rect.right(), shape[1]),
                          min(rect.bottom(), shape[0]))


def face_locations(img, number_of_times_to_upsample=1, model="hog"):
    image_shape = img.shape
    if model == "cnn":
        return [_rect_to_bound(face.rect, image_shape) for face in cnn_face_detector(img, number_of_times_to_upsample)]
    else:
        return [_rect_to_bound(rect, image_shape) for rect in face_detector(img, number_of_times_to_upsample)]


def face_location_as_css(location):
    return location.top(), location.right(), location.bottom(), location.left()


def face_locations_as_css(locations):
    return [face_location_as_css(rect) for rect in locations]


def face_location_from_css(css):
    return dlib.rectangle(css[3], css[0], css[1], css[2])


def face_locations_from_css(css_list):
    return [face_location_from_css(css) for css in css_list]


def face_landmarks(face_image, face_locations_=None, model='small'):
    if face_locations_ is None:
        face_locations_ = face_locations(face_image)

    pose_predictor = pose_predictor_5_point if model == "small" else pose_predictor_68_point
    return [pose_predictor(face_image, location) for location in face_locations_]


def face_landmarks_detail(face_image, face_locations_=None):
    landmarks = face_landmarks(face_image, face_locations_, 'large')
    landmarks_as_tuples = [[(p.x, p.y) for p in landmark.parts()] for landmark in landmarks]

    return [{
        "chin": points[0:17],
        "left_eyebrow": points[17:22],
        "right_eyebrow": points[22:27],
        "nose_bridge": points[27:31],
        "nose_tip": points[31:36],
        "left_eye": points[36:42],
        "right_eye": points[42:48],
        "top_lip": points[48:55] + [points[64]] + [points[63]] + [points[62]] + [points[61]] + [points[60]],
        "bottom_lip": points[54:60] + [points[48]] + [points[60]] + [points[67]] + [points[66]] + [points[65]] + [
            points[64]]
    } for points in landmarks_as_tuples]


def face_encodings(face_image, face_landmarks_, num_jitters=1, *, version=1):
    if version == 1:
        face_encoder = face_encoder_v1_1
    else:
        face_encoder = face_encoder_v1_2

    return [np.array(face_encoder.compute_face_descriptor(face_image, lm, num_jitters)) for lm in face_landmarks_]


class ImageMixin:

    @staticmethod
    def _load_image(data_or_io, mode='RGB'):
        if type(data_or_io) == BufferedReader:
            data_or_io.seek(0)
            data_or_io = data_or_io.read()
        elif type(data_or_io) == str:
            data_or_io = b64decode(data_or_io)

        with BytesIO(data_or_io) as io_:
            image = Image.open(io_)
            image_format = image.format
            if mode:
                image = image.convert(mode)
            image_array = np.array(image)
        return image_array, image_format, data_or_io

    @staticmethod
    def _max_face_index(locations):
        max_, index = 0, -1
        for i, l in enumerate(locations):
            size_ = l.width() * l.height()
            if size_ > max_:
                max_ = size_
                index = i
        return index


class FaceImage(ImageMixin):

    def __init__(self, id_, data_or_io, aligned, upsample_times, model, num_jitters, resnet_model_version=1):
        self.id = id_
        image_array, self.format, self._raw_data = self._load_image(data_or_io)
        if aligned:
            self._locations = [face_location_from_css((0, image_array.shape[1], image_array.shape[0], 0))]
        else:
            self._locations = face_locations(image_array, upsample_times, model)

        if self._locations:
            self.encodings = face_encodings(image_array, face_landmarks(image_array, self._locations), num_jitters,
                                            version=resnet_model_version)
            self.number_of_times_to_upsample = upsample_times
            self.model = model
            self.num_jitters = num_jitters

    @property
    def has_face(self):
        return bool(self._locations)

    @property
    def locations(self):
        return face_locations_as_css(self._locations)

    def max_one(self):
        if not self.has_face:
            return self

        index = self._max_face_index(self._locations)

        pi = copy(self)
        pi._locations = [self._locations[index]]
        pi.encodings = [self.encodings[index]]
        return pi


class FaceLocation(ImageMixin):
    def __init__(self, data_or_io, upsample_times, model):
        self.landmark = None
        image_array, _, _ = self._load_image(data_or_io)

        locations = face_locations(image_array, upsample_times, model)
        if locations and len(locations) > 0:
            index = self._max_face_index(locations)
            location = locations[index]

            landmarks = face_landmarks_detail(image_array, [location])
            if landmarks and len(landmarks) > 0:
                self.landmark = landmarks[0]

    def __repr__(self):
        return str(self.landmark) if self else ''

    def __bool__(self):
        return self.landmark is not None

    def __dict__(self):
        return self.landmark if self else {}

    def __getattr__(self, name):
        if self:
            return self.landmark[name]
