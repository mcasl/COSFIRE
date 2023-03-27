import unittest
import numpy as np
from numpy.testing import assert_almost_equal
import logging
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import cv2

from cosfire.base import (Cosfire,
                          CosfireCircularGaborTuple,
                          GaborParameters,
                          π,
                          _Circular_Gabor__compute_bank_of_responses,
                          _Circular_Gabor__fit_Sf
                          )

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)

preset_1 = dict(strategy_name="Circular Gabor",
                center_x=166,
                center_y=96,
                rho_list=[0, 12],
                eta=np.pi / 8,
                t1=0.99,
                t2=0.75,
                t3=0.9,
                filter_parameters=GaborParameters(ksize=(10, 10),
                                                  σ=5,
                                                  θ=np.array([0, π / 2]),
                                                  λ=np.array([12]),
                                                  γ=0.5,
                                                  ψ=π,
                                                  ktype=cv2.CV_32F),
                sigma0=0.67,
                alpha=0.04,
                reflection_invariant=0,
                scale_invariant=[1],
                rotation_invariant=[0]
                )

preset_2 = dict(strategy_name="Circular Gabor",
                center_coordinate_x=166,
                center_coordinate_y=96,
                rho_list=[0, 12],
                eta=π / 8,
                t1=0.99,
                t2=0.75,
                t3=0.9,
                filter_parameters=[(10, 10),
                                   5,
                                   np.array([0, π / 2]),
                                   np.array([12]),
                                   0.5,
                                   π],
                sigma0=0.67,
                alpha=0.04,
                reflection_invariant=True,
                scale_invariant=[1],
                rotation_invariant=[0]
                )

preset_3 = dict(strategy_name="Circular Gabor",
                center_coordinate_x=24,
                center_coordinate_y=28,
                rho_list=[0, 2, 4, 7, 10, 13, 16, 20, 25],
                eta=π / 8,
                t1=0.1,
                t2=0.75,
                t3=0.99,
                filter_parameters=[(10, 10),
                                   0.8,
                                   np.array([(i * π) / 8.0 for i in range(16)]),
                                   np.array([4]),
                                   0.5,
                                   π / 2],
                sigma0=0.67,
                alpha=0.04,
                reflection_invariant=False,
                scale_invariant=[1],
                rotation_invariant=[0]
                )

preset_6 = dict(strategy_name="Circular Gabor",
                center_coordinate_x=17,
                center_coordinate_y=17,
                rho_list=[0, 3, 8],
                eta=π / 8,
                t1=0.05,
                t2=0.75,
                t3=0.3,
                filter_parameters=[np.array([(i * π) / 8.0 for i in range(16)]),
                                   np.array([2 * np.sqrt(2)]),
                                   (5, 5),
                                   0.8,
                                   0.5,
                                   π / 2],
                sigma0=0.83,
                alpha=0.1,
                reflection_invariant=False,
                scale_invariant=[1],
                rotation_invariant=[0]
                )


class TestCosfire(unittest.TestCase):
    def setUp(self):
        self.pattern = 'media/patron.1'

    def test_cosfire__preset_1(self):
        im1 = cv2.imread("media/patron1.jpg", 0)
        im2 = cv2.imread("media/prueba1.jpg", 0)
        for _ in range(1):
            a = Cosfire(**preset_1)
            a.fit(im1)
            r = a.transform(im2)
        # viewPattern(r, im2, 20)
        # cv2.imshow("Response", r)
        expected = np.load('media/test_cosfire__preset_1_expected.npy')
        assert_almost_equal(r, expected)

    def test_shift_responses(self):
        specific_image = np.array([[3, 3, 3, 3, 3],
                                   [3, 2, 2, 2, 3],
                                   [3, 2, 1, 2, 3],
                                   [3, 2, 2, 2, 3],
                                   [3, 3, 3, 3, 3]], dtype=np.float32)
        tupla_collection = [CosfireCircularGaborTuple(ρ=1, ϕ=0, λ=0.1, θ=0),
                            CosfireCircularGaborTuple(ρ=1, ϕ=π * 0.5, λ=0.1, θ=0),
                            CosfireCircularGaborTuple(ρ=1, ϕ=π, λ=0.1, θ=0),
                            CosfireCircularGaborTuple(ρ=1, ϕ=π * 1.5, λ=0.1, θ=0),
                            ]
        response_collection = {tupla: specific_image for tupla in tupla_collection}
        cosfire_filter = Cosfire(**preset_1)
        result_collection = {key: cosfire_filter.shift_responses({key: response})[key]
                             for key, response in response_collection.items()}
        expected_collection = {
            CosfireCircularGaborTuple(ρ=1, ϕ=0, λ=0.1, θ=0): np.array([[3, 3, 3, 3, 0],
                                                                       [2, 2, 2, 3, 0],
                                                                       [2, 1, 2, 3, 0],
                                                                       [2, 2, 2, 3, 0],
                                                                       [3, 3, 3, 3, 0]], dtype=np.float32),
            CosfireCircularGaborTuple(ρ=1, ϕ=π * 0.5, λ=0.1, θ=0): np.array([[0., 0., 0., 0., 0.],
                                                                             [3., 3., 3., 3., 3.],
                                                                             [3., 2., 2., 2., 3.],
                                                                             [3., 2., 1., 2., 3.],
                                                                             [3., 2., 2., 2., 3.]],
                                                                            dtype=np.float32),
            CosfireCircularGaborTuple(ρ=1, ϕ=π, λ=0.1, θ=0): np.array([[0., 3., 3., 3., 3.],
                                                                       [0., 3., 2., 2., 2.],
                                                                       [0., 3., 2., 1., 2.],
                                                                       [0., 3., 2., 2., 2.],
                                                                       [0., 3., 3., 3., 3.]], dtype=np.float32),
            CosfireCircularGaborTuple(ρ=1, ϕ=π * 1.5, λ=0.1, θ=0): np.array([[3., 2., 2., 2., 3.],
                                                                             [3., 2., 1., 2., 3.],
                                                                             [3., 2., 2., 2., 3.],
                                                                             [3., 3., 3., 3., 3.],
                                                                             [0., 0., 0., 0., 0.]],
                                                                            dtype=np.float32), }

        assert_almost_equal(result_collection[CosfireCircularGaborTuple(ρ=1, ϕ=π, λ=0.1, θ=0)],
                            expected_collection[CosfireCircularGaborTuple(ρ=1, ϕ=π, λ=0.1, θ=0)])

        assert_almost_equal(result_collection[CosfireCircularGaborTuple(ρ=1, ϕ=0, λ=0.1, θ=0)],
                            expected_collection[CosfireCircularGaborTuple(ρ=1, ϕ=0, λ=0.1, θ=0)])

        assert_almost_equal(result_collection[CosfireCircularGaborTuple(ρ=1, ϕ=π * 0.5, λ=0.1, θ=0)],
                            expected_collection[CosfireCircularGaborTuple(ρ=1, ϕ=π * 0.5, λ=0.1, θ=0)])

        assert_almost_equal(result_collection[CosfireCircularGaborTuple(ρ=1, ϕ=π * 1.5, λ=0.1, θ=0)],
                            expected_collection[CosfireCircularGaborTuple(ρ=1, ϕ=π * 1.5, λ=0.1, θ=0)])

        # for key, result in result_collection.items():
        #     assert_almost_equal(result, expected_collection[key])

    def test_threshold_prototype_bank_of_responses(self):
        self._maximum_response = max(
            value.max() for key, value in self._prototype_bank.items()
        )
        [cv2.threshold(src=image, dst=image, thresh=threshold * self._maximum_response,
                       maxval=self._maximum_response, type=cv2.THRESH_TOZERO)
         for key, image in self._prototype_bank.items()]  # Desired collateral effect: modify self.responses_map
        return


class TestCosfireCircularGabor(unittest.TestCase):
    def setUp(self):
        self.pattern = np.zeros((256, 256))
        cv2.rectangle(img=self.pattern, pt1=(50, 100), pt2=(100, 106), color=255, thickness=-1)
        cv2.rectangle(img=self.pattern, pt1=(100, 100), pt2=(97, 50), color=255, thickness=-1)

        self.cosfire = Cosfire(strategy_name='Circular Gabor',
                               center_x=100,
                               center_y=100,
                               rho_list=range(0, 100, 10),
                               t1=0.99,
                               t2=0.75,
                               t3=0.9,
                               filter_parameters=GaborParameters(ksize=(10, 10), σ=5,
                                                                 θ=np.linspace(start=0, stop=π, num=30,
                                                                               endpoint=False),
                                                                 λ=np.linspace(start=7, stop=8, num=10,
                                                                               endpoint=False),
                                                                 γ=0.5, ψ=π, ktype=cv2.CV_32F),
                               sigma0=0.67,
                               alpha=0.04,
                               reflection_invariant=0,
                               scale_invariant=[1],
                               rotation_invariant=[0]
                               )

    def test__cosfire_process(self):
        self.cosfire.fit(self.pattern)
        expected = [CosfireCircularGaborTuple(λ=7.5, θ=0.0, ρ=10, φ=1.3089969389957472),
                    CosfireCircularGaborTuple(λ=7.5, θ=0.0, ρ=10, φ=2.0420352248333655),
                    CosfireCircularGaborTuple(λ=7.5, θ=1.5707963267948966, ρ=10, φ=2.9845130209103035),
                    CosfireCircularGaborTuple(λ=7.5, θ=1.5707963267948966, ρ=10, φ=4.171336912266447),
                    CosfireCircularGaborTuple(λ=7.5, θ=0.0, ρ=20, φ=1.7976891295541595),
                    CosfireCircularGaborTuple(λ=7.5, θ=1.5707963267948966, ρ=20, φ=3.07177948351002),
                    CosfireCircularGaborTuple(λ=7.5, θ=1.5707963267948966, ρ=20, φ=3.5779249665883754),
                    CosfireCircularGaborTuple(λ=7.5, θ=0.0, ρ=30, φ=1.710422666954443),
                    CosfireCircularGaborTuple(λ=7.5, θ=1.5707963267948966, ρ=30, φ=3.420845333908886),
                    CosfireCircularGaborTuple(λ=7.5, θ=0.0, ρ=40, φ=1.6755160819145565),
                    CosfireCircularGaborTuple(λ=7.5, θ=1.5707963267948966, ρ=40, φ=3.351032163829113)]

        result = self.cosfire._Sf
        self.assertEqual(result, expected)

    def test__Circular_Gabor__compute_bank_of_responses__no_kwargs(self):
        self.pattern = np.zeros((256, 256))
        self.pattern = cv2.circle(img=self.pattern, center=(100,100), radius=75, color=255)
        cosfire = Cosfire(strategy_name='Circular Gabor',
                          center_x=100,
                          center_y=100,
                          rho_list=range(0, 50, 10),
                          t1=0.99,
                          t2=0.75,
                          t3=0.9,
                          filter_parameters=GaborParameters(ksize=(10, 10), σ=5,
                                                            θ=np.linspace(start=0, stop=π, num=5,
                                                                          endpoint=True),
                                                            λ=[12],#np.linspace(start=1, stop=100, num=100, endpoint=True),
                                                            γ=0.25, ψ=0, ktype=cv2.CV_32F),
                          sigma0=0.67,
                          alpha=0.04,
                          reflection_invariant=0,
                          scale_invariant=[1],
                          rotation_invariant=[0]
                          )

        results_bank = _Circular_Gabor__compute_bank_of_responses(cosfire, self.pattern)

        generating_results = False
        if generating_results:
            #The first time I used this to generate expected results
            np.save('media/test__Circular_Gabor__compute_bank_of_responses__no_kwargs__pattern.npy', self.pattern)
            with PdfPages('test__Circular_Gabor__compute_bank_of_responses.pdf') as pdf:
                for key, value in results_bank.items():
                    file_name = f'media/test__Circular_Gabor__compute_bank_of_responses__no_kwargs__{str(key)}.npy'
                    np.save(file_name, value)
                    fig = plt.figure()
                    plt.imshow(value, cmap='cool')
                    plt.title(key)
                    pdf.savefig(fig)

        for key in results_bank:
            file_name = f'media/test__Circular_Gabor__compute_bank_of_responses__no_kwargs__{str(key)}.npy'
            expected = np.load(file_name)
            result = results_bank[key]
            assert_almost_equal(result, expected)

    def test__threshold_prototype_bank_of_responses(self):
        self.pattern = np.zeros((256, 256))
        self.pattern = cv2.circle(img=self.pattern, center=(100, 100), radius=75, color=255)
        cosfire = Cosfire(strategy_name='Circular Gabor',
                          center_x=100,
                          center_y=100,
                          rho_list=range(0, 50, 10),
                          t1=0.99,
                          t2=0.75,
                          t3=0.9,
                          filter_parameters=GaborParameters(ksize=(10, 10), σ=5,
                                                            θ=np.linspace(start=0, stop=π, num=5,
                                                                          endpoint=True),
                                                            λ=[12],
                                                            # np.linspace(start=1, stop=100, num=100, endpoint=True),
                                                            γ=0.25, ψ=0, ktype=cv2.CV_32F),
                          sigma0=0.67,
                          alpha=0.04,
                          reflection_invariant=0,
                          scale_invariant=[1],
                          rotation_invariant=[0]
                          )

        cosfire._prototype_bank = _Circular_Gabor__compute_bank_of_responses(cosfire, self.pattern)
        cosfire.threshold_prototype_bank_of_responses(threshold=0.75)

        generating_results = False
        if generating_results:
            # The first time I used this to generate expected results
            np.save('media/test__threshold_prototype_bank_of_responses__pattern.npy', self.pattern)
            with PdfPages('test__threshold_prototype_bank_of_responses.pdf') as pdf:
                for key, value in cosfire._prototype_bank.items():
                    file_name = f'media/test__threshold_prototype_bank_of_responses_{str(key)}.npy'
                    np.save(file_name, value)
                    fig = plt.figure()
                    plt.imshow(value, cmap='cool')
                    plt.title(key)
                    pdf.savefig(fig)

        for key in cosfire._prototype_bank:
            file_name = f'media/test__threshold_prototype_bank_of_responses_{str(key)}.npy'
            expected = np.load(file_name)
            result = cosfire._prototype_bank[key]
            assert_almost_equal(result, expected)

    def test__Circular_Gabor__fit_Sf(self):
        self.pattern = np.zeros((256, 256))
        self.pattern = cv2.circle(img=self.pattern, center=(100, 100), radius=75, color=255)
        cosfire = Cosfire(strategy_name='Circular Gabor',
                          center_x=100,
                          center_y=100,
                          rho_list=(0, 75, 100),
                          t1=0.99,
                          t2=0.75,
                          t3=0.9,
                          filter_parameters=GaborParameters(ksize=(10, 10), σ=5,
                                                            θ=np.linspace(start=0, stop=π, num=5,
                                                                          endpoint=True),
                                                            λ=[12],
                                                            # np.linspace(start=1, stop=100, num=100, endpoint=True),
                                                            γ=0.25, ψ=0, ktype=cv2.CV_32F),
                          sigma0=0.67,
                          alpha=0.04,
                          reflection_invariant=0,
                          scale_invariant=[0],
                          rotation_invariant=[0]
                          )

        cosfire._prototype_bank = _Circular_Gabor__compute_bank_of_responses(cosfire, self.pattern)
        cosfire.threshold_prototype_bank_of_responses(threshold=0.85)
        operator = _Circular_Gabor__fit_Sf(cosfire)
        self.assertEqual(operator,[CosfireCircularGaborTuple(λ=12, θ=0.0,                ρ=75, φ=0.06981317007977318),
                                   CosfireCircularGaborTuple(λ=12, θ=3.141592653589793,  ρ=75, φ=0.06981317007977318),
                                   CosfireCircularGaborTuple(λ=12, θ=2.356194490192345,  ρ=75, φ=0.6457718232379019 ),
                                   CosfireCircularGaborTuple(λ=12, θ=2.356194490192345,  ρ=75, φ=4.066617157146788  ),
                                   CosfireCircularGaborTuple(λ=12, θ=1.5707963267948966, ρ=75, φ=4.625122517784973  ),
                                   CosfireCircularGaborTuple(λ=12, θ=0.7853981633974483, ρ=75, φ=5.480333851262195  ),
                                   CosfireCircularGaborTuple(λ=12, θ=0.0,                ρ=75, φ=6.178465552059927  ),
                                   CosfireCircularGaborTuple(λ=12, θ=3.141592653589793,  ρ=75, φ=6.178465552059927  )])
