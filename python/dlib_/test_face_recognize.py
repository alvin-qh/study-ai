from unittest import TestCase


class TestModelLoad(TestCase):

    def test_model_load(self):
        from .face_recognize import face_detector, cnn_face_detector, pose_predictor_5_point, pose_predictor_68_point, \
            face_encoder_v1_1, face_encoder_v1_2

        assert face_detector
        assert cnn_face_detector
        assert pose_predictor_5_point
        assert pose_predictor_68_point
        assert face_encoder_v1_1
        assert face_encoder_v1_2
