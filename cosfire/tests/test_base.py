import unittest
from pathlib import Path

import cv2
import numpy as np
from numpy.testing import assert_almost_equal

from cosfire.base import (Cosfire,
                          CosfireCircularGaborTuple,
                          DoGParameters,
                          GaborParameters,
                          Pixel,
                          π,
                          _Circular_Gabor__blur_gaussian,
                          _Circular_Gabor__compute_bank_of_responses,
                          _Circular_Gabor__fit_Sf,
                          )

MEDIA = Path(__file__).parent / 'media'

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


def circle_pattern():
    pattern = np.zeros((256, 256))
    cv2.circle(img=pattern, center=(100, 100), radius=75, color=255)
    return pattern


def circle_gabor_cosfire(**overrides):
    parameters = dict(strategy_name='Circular Gabor',
                      center_x=100,
                      center_y=100,
                      rho_list=range(0, 50, 10),
                      t1=0.99,
                      t2=0.75,
                      t3=0.9,
                      filter_parameters=GaborParameters(ksize=(10, 10), σ=5,
                                                        θ=np.linspace(start=0, stop=π, num=5, endpoint=True),
                                                        λ=[12],
                                                        γ=0.25, ψ=0, ktype=cv2.CV_32F),
                      sigma0=0.67,
                      alpha=0.04,
                      reflection_invariant=0,
                      scale_invariant=[1],
                      rotation_invariant=[0])
    parameters.update(overrides)
    return Cosfire(**parameters)


class TestCosfire(unittest.TestCase):
    def test_cosfire__preset_1(self):
        im1 = cv2.imread(str(MEDIA / "patron1.jpg"), 0)
        im2 = cv2.imread(str(MEDIA / "prueba1.jpg"), 0)
        self.assertIsNotNone(im1)
        self.assertIsNotNone(im2)
        cosfire = Cosfire(**preset_1).fit(im1)

        response = cosfire.transform(im2)
        self.assertEqual(response.shape, im2.shape)
        self.assertTrue(np.isfinite(response).all())
        self.assertGreaterEqual(response.min(), 0)

        # applied to its own prototype the filter must respond around the prototype center
        self_response = cosfire.transform(im1)
        row, column = np.unravel_index(self_response.argmax(), self_response.shape)
        self.assertLessEqual(abs(row - preset_1['center_x']), 3)
        self.assertLessEqual(abs(column - preset_1['center_y']), 3)

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
        cosfire = circle_gabor_cosfire()
        results_bank = _Circular_Gabor__compute_bank_of_responses(cosfire, circle_pattern())

        self.assertEqual(len(results_bank), 5)  # 5 orientations x 1 wavelength
        for key, response in results_bank.items():
            self.assertEqual(key.λ, 12)
            self.assertEqual(response.shape, (256, 256))
            self.assertEqual(response.dtype, np.float32)
            self.assertTrue(np.isfinite(response).all())

    def test__threshold_prototype_bank_of_responses(self):
        cosfire = circle_gabor_cosfire()
        cosfire._prototype_bank = _Circular_Gabor__compute_bank_of_responses(cosfire, circle_pattern())
        original_maximum = max(value.max() for value in cosfire._prototype_bank.values())

        cosfire.threshold_prototype_bank_of_responses(threshold=0.75)

        self.assertAlmostEqual(cosfire._maximum_response, original_maximum, places=4)
        for response in cosfire._prototype_bank.values():
            surviving = response[response != 0]
            self.assertTrue((surviving > 0.75 * original_maximum).all())

    def test__Circular_Gabor__fit_Sf(self):
        # Hardware/OpenCV-version independent: every tuple must point at a pixel of the circle
        pattern = circle_pattern()
        cosfire = circle_gabor_cosfire(rho_list=(0, 75, 100))
        cosfire._prototype_bank = _Circular_Gabor__compute_bank_of_responses(cosfire, pattern)
        cosfire.threshold_prototype_bank_of_responses(threshold=0.85)

        operator = _Circular_Gabor__fit_Sf(cosfire)

        self.assertGreater(len(operator), 0)
        circle_points = np.argwhere(pattern > 0)
        for tupla in operator:
            self.assertEqual(tupla.ρ, 75)  # rho=0 is background and rho=100 lies outside the circle
            self.assertEqual(tupla.λ, 12)
            row = cosfire.center_x - tupla.ρ * np.sin(tupla.ϕ)
            column = cosfire.center_y + tupla.ρ * np.cos(tupla.ϕ)
            distance = np.sqrt(((circle_points - [row, column]) ** 2).sum(axis=1)).min()
            self.assertLess(distance, 2.0)

    def test__default_invariances_do_not_produce_an_empty_operator(self):
        # Regression: scale_invariant/rotation_invariant defaulted to [] and transform crashed on max([])
        parameters = dict(center_x=100, center_y=100, rho_list=(0, 75), t1=0.5, t2=0.5, t3=0.5,
                          filter_parameters=circle_gabor_cosfire().filter_parameters,
                          sigma0=0.67, alpha=0.04)
        cosfire = Cosfire(**parameters)
        self.assertEqual(cosfire.rotation_invariant, [0])
        self.assertEqual(cosfire.scale_invariant, [1])

        pattern = circle_pattern()
        response = cosfire.fit(pattern).transform(pattern)

        self.assertGreater(response.max(), 0)

    def test__fit_returns_self(self):
        cosfire = circle_gabor_cosfire()
        self.assertIs(cosfire.fit(circle_pattern()), cosfire)

    def test__reflection_invariant_accepts_booleans(self):
        pattern = circle_pattern()
        cosfire = circle_gabor_cosfire(rho_list=(0, 75), t1=0.5, t2=0.5, t3=0.5, reflection_invariant=True)
        response = cosfire.fit(pattern).transform(pattern)
        self.assertGreater(response.max(), 0)

    def test__uint8_images_are_accepted(self):
        pattern = circle_pattern().astype(np.uint8)
        cosfire = circle_gabor_cosfire(rho_list=(0, 75), t1=0.5, t2=0.5, t3=0.5)
        response = cosfire.fit(pattern).transform(pattern)
        self.assertGreater(response.max(), 0)

    def test__blur_gaussian_does_not_truncate_large_sigma(self):
        # Regression: a fixed 9x9 kernel (normalised anyway) cut a sigma=8.67 gaussian at +-4 pixels
        cosfire = circle_gabor_cosfire()
        tupla = CosfireCircularGaborTuple(λ=12, θ=0., ρ=200, ϕ=0.)  # sigma = 0.04 * 200 + 0.67
        cosfire._Sf_invariant = [tupla]
        impulse = np.zeros((101, 101), dtype=np.float32)
        impulse[50, 50] = 1

        blurred = _Circular_Gabor__blur_gaussian(cosfire, {(0., 12): impulse})[tupla]

        self.assertAlmostEqual(float(blurred.sum()), 1.0, places=3)
        self.assertGreater(blurred[50, 60], 0)  # ten pixels away from the impulse, beyond the old 9x9 support

    def test__pixel_outside_image(self):
        image = np.zeros((4, 5))
        self.assertTrue(Pixel(row=3, column=4).is_inside(image))
        self.assertFalse(Pixel(row=-1, column=0).is_inside(image))
        self.assertFalse(Pixel(row=0, column=5).is_inside(image))


class TestCosfireCircularDoG(unittest.TestCase):
    def setUp(self):
        self.pattern = circle_pattern()

    def make(self, **overrides):
        parameters = dict(strategy_name='Circular DoG',
                          center_x=100,
                          center_y=100,
                          rho_list=(0, 75),
                          t1=0.3,
                          t2=0.3,
                          t3=0.5,
                          filter_parameters=DoGParameters(σ=(2., 4.)),
                          sigma0=0.67,
                          alpha=0.04,
                          reflection_invariant=False,
                          scale_invariant=[1],
                          rotation_invariant=[0])
        parameters.update(overrides)
        return Cosfire(**parameters)

    def test__fit_and_transform_detect_the_prototype_center(self):
        # Regression: fit_Sf used a non-existent attribute and the DoG strategy always crashed
        cosfire = self.make().fit(self.pattern)
        self.assertGreater(len(cosfire._Sf), 0)

        response = cosfire.transform(self.pattern)

        row, column = np.unravel_index(response.argmax(), response.shape)
        self.assertLessEqual(abs(row - 100), 2)
        self.assertLessEqual(abs(column - 100), 2)

    def test__reflection_and_multiple_invariances(self):
        cosfire = self.make(reflection_invariant=True, rotation_invariant=[0, π / 2], scale_invariant=[1, 2])
        response = cosfire.fit(self.pattern).transform(self.pattern)
        self.assertEqual(response.shape, self.pattern.shape)
        self.assertGreater(response.max(), 0)

    def test__uint8_images_do_not_wrap_around(self):
        cosfire = self.make()
        bank = cosfire.compute_bank_of_responses(self.pattern.astype(np.uint8))
        self.assertTrue(all(response.dtype == np.float32 for response in bank.values()))
        self.assertTrue(all(response.min() < 0 for response in bank.values()))  # DoG has negative lobes


if __name__ == '__main__':
    unittest.main()
