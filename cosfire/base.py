#   Keypoint detection by a COSFIRE filter
#   Refactored from Daniel Acosta 2017 code
#
#   COSFIRE is an algorithm created by George Azzopardi and Nicolai Petkov,
#   for more details view original paper:
#   George Azzopardi and Nicolai Petkov, "Trainable COSFIRE filters for
#   keypoint detection and pattern recognition", IEEE Transactions on Pattern 
#   Analysis and Machine Intelligence, vol. 35(2), pp. 490-503, 2013.

import numpy as np
import cv2
import peakutils
from typing import NamedTuple
import itertools
from math import floor, cos, sin

π = np.pi


class Pixel(NamedTuple):
    row: int
    column: int

    def is_inside(self, image):
        rows, cols = image.shape
        return (0 <= self.row < rows) and (0 <= self.column < cols)

    def maximum_response(self, bank):
        for image in bank.values():
            some_response_map = image
            break  # I just want one image to check the size

        return (
            max((response[self] for response in bank.values()))
            if self.is_inside(some_response_map)
            else 0
        )

    def evaluate_bank(self, bank):
        for value in bank.values():
            yield value[self]


class GaborParameters(NamedTuple):
    ksize: tuple
    σ: float
    θ: float
    λ: float
    γ: float
    ψ: float = π * 0.5
    ktype: int = cv2.CV_32F


class GaborKey(NamedTuple):
    θ: float
    λ: float


class CosfireCircularGaborTuple(NamedTuple):
    λ: float
    θ: float
    ρ: float
    ϕ: float


class CosfireCircularDoGTuple(NamedTuple):
    ρ: float
    ϕ: float
    σ: float


class Cosfire:
    def __init__(self,
                 strategy_name='Circular Gabor',
                 center_x=0,
                 center_y=0,
                 rho_list=None,
                 eta=0,
                 t1=0,
                 filter_parameters=None,
                 t2=0,
                 alpha=0,
                 sigma0=0,
                 t3=0,
                 reflection_invariant=False,
                 rotation_invariant=None,
                 scale_invariant=None,
                 ddepth=cv2.CV_32F):
        self.strategy_name = strategy_name
        self.center_x = center_x
        self.center_y = center_y
        self.ρ_list = [] if rho_list is None else rho_list
        self.η = eta  # TODO: unused!!!
        self.threshold_1 = t1
        self.filter_parameters = [] if filter_parameters is None else filter_parameters  # Parameters of filter
        self.threshold_2 = t2
        self.alpha = alpha
        self.σ0 = sigma0
        self.threshold_3 = t3
        self.reflection_invariant = reflection_invariant
        self.scale_invariant = [] if scale_invariant is None else scale_invariant
        self.rotation_invariant = [] if rotation_invariant is None else rotation_invariant
        self.ddepth = ddepth

        strategies = strategies_dictionary[self.strategy_name]
        self._compute_bank_of_responses = strategies['compute_bank_of_responses']
        self._i_scale_cosfire = strategies['i_scale_cosfire']
        self._i_rotation_cosfire = strategies['i_rotation_cosfire']
        self._i_reflection_cosfire = strategies['i_reflection_cosfire']
        self._blur_gaussian = strategies['blur_gaussian']
        self._compute_tuples = strategies['compute_tuples']
        self._fit_Sf = strategies['fit_Sf']

        self._responses_to_image = {}
        self._Sf = []  # Struct of filter COSFIRE (S_f)
        self._prototype_bank = {}  # Bank of responses pattern Image
        self._maximum_response = 0
        self._Sf_invariant = []  # operator invariant to rotation, escala and reflection

    # 1-Configuration COSFIRE Filter
    def fit(self, X, **kwargs):
        self.prototype_image = X
        self._prototype_bank = self.compute_bank_of_responses(self.prototype_image)  # 1.1
        self.threshold_prototype_bank_of_responses(self.threshold_1)  # 1.2
        # Manolo: This is an optimization possibility but requires to modify the iteration on θ, λ during the calculation of the pixel responses
        # self._prototype_bank = {key: value for key, value in self._prototype_bank.items() if value.max() > self._maximum_response * self.threshold_2}
        self._Sf = self.fit_Sf()  # 1.3

    # 2-Apply the COSFIRE filter
    def transform(self, X, **kwargs):
        input_image = X
        self._Sf_invariant = []
        input_image_bank_of_responses = self.compute_tuples(input_image)  # 2.1
        input_image_bank_of_responses = self.shift_responses(input_image_bank_of_responses)  # 2.2
        output_image = self.i_reflection_cosfire(input_image, self._Sf, input_image_bank_of_responses)  # 2.3
        maximum_output = output_image.max()
        cv2.threshold(src=output_image, dst=output_image, thresh=self.threshold_3 * maximum_output,
                      maxval=maximum_output,
                      type=cv2.THRESH_TOZERO)
        return output_image

    # (1.1) Get response filter
    def compute_bank_of_responses(self, image, **kwargs):
        return self._compute_bank_of_responses(self, image, **kwargs)

    # (1.2) Suppress Responses
    def threshold_prototype_bank_of_responses(self, threshold, **kwargs):
        self._maximum_response = max(
            value.max() for key, value in self._prototype_bank.items()
        )
        [cv2.threshold(src=image, dst=image, thresh=threshold * self._maximum_response,
                       maxval=self._maximum_response, type=cv2.THRESH_TOZERO)
         for key, image in self._prototype_bank.items()]  # Desired collateral effect: modify self.responses_map
        return

    # (1.3) Get descriptor set (Sf)
    def fit_Sf(self, **kwargs):
        return self._fit_Sf(self, **kwargs)

    # 2.1 For each tuple in the set Sf compute response
    def compute_tuples(self, inputImage, **kwargs):
        return self._compute_tuples(self, inputImage, **kwargs)

    # (2.1.1)Blurr
    def blur_gaussian(self, bankFilters, **kwargs):
        return self._blur_gaussian(self, bankFilters, **kwargs)

    # (2.2) Shift
    def shift_responses(self, responses, **kwargs):
        response_maps = dict.fromkeys(responses)
        for tupla, responses in responses.items():
            rows, cols = responses.shape
            x = -tupla.ρ * cos(tupla.ϕ)
            y = tupla.ρ * sin(tupla.ϕ)
            M = np.float32([[1, 0, x],
                            [0, 1, y]])
            dst = cv2.warpAffine(responses, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            response_maps[tupla] = dst
        return response_maps

    # (2.3) invariant under reflection
    def i_reflection_cosfire(self, image, operator, response_bank, **kwargs):
        return self._i_reflection_cosfire(self, image, operator, response_bank, **kwargs)

    # (2.4) invariant to rotation
    def i_rotation_cosfire(self, image, operator, response_bank, **kwargs):
        return self._i_rotation_cosfire(self, image, operator, response_bank, **kwargs)

    # (2.5) invariant to scale
    def i_scale_cosfire(self, image, operator, response_bank, **kwargs):
        return self._i_scale_cosfire(self, image, operator, response_bank, **kwargs)

    # Compute response
    def average_response(self, bank, operator, **kwargs):
        rows, cols = bank[operator[0]].shape
        resp = np.ones((rows, cols))
        ρ_maximum = max(tupla.ρ for tupla in operator)
        tuple_weight_σ = np.sqrt(-((ρ_maximum ** 2) / (2 * np.log(0.5))))

        sum_of_scalar_power = 0
        for tupla in operator:
            scalar_power = np.exp(-(tupla.ρ ** 2) / (2 * tuple_weight_σ ** 2))
            sum_of_scalar_power += scalar_power
            resp *= np.float_power(bank[tupla], scalar_power)
        resp = np.power(resp, 1.0 / sum_of_scalar_power)
        return resp


def _Circular_Gabor__compute_bank_of_responses(self, image, **kwargs):
    ksize, σ, θ_array, λ_array, γ, ψ, ktype = self.filter_parameters
    if 'θ_array' in kwargs:
        θ_array = kwargs['θ_array']
    if 'λ_array' in kwargs:
        λ_array = kwargs['λ_array']

    return {
        GaborKey(θ=θ, λ=λ): cv2.filter2D(
            src=image,
            ddepth=self.ddepth,
            kernel=cv2.getGaborKernel(
                ksize=ksize,
                sigma=σ,
                theta=θ,
                lambd=λ,
                gamma=γ,
                psi=ψ,
                ktype=ktype,
            ),
        )
        for θ, λ in itertools.product(θ_array, λ_array)
    }


def _Circular_DoG__compute_bank_of_responses(self, image, **kwargs):
    response = {}
    for σ in self.filter_parameters.σ:
        g1 = cv2.GaussianBlur(image, (3, 3), σ)
        g2 = cv2.GaussianBlur(image, (3, 3), 0.5 * σ)
        r = g1 - g2
        response[σ] = r
    return response


def _Circular_Gabor__i_scale_cosfire(self, image, operator, response_bank, **kwargs):
    maximum_output = np.zeros_like(image)
    for scale in self.scale_invariant:
        operatorEscala = [
            CosfireCircularGaborTuple(λ=tupla.λ * scale, θ=tupla.θ, ρ=tupla.ρ * scale, ϕ=tupla.ϕ) for tupla in operator]
        output = self.average_response(response_bank, operatorEscala, **kwargs)
        maximum_output = np.maximum(output, maximum_output)
    return maximum_output


def _Circular_DoG__i_scale_cosfire(self, image, operator, respBank, **kwargs):
    maximum_output = np.zeros_like(image)
    for scale in self.scale_invariant:
        operatorEscala = [CosfireCircularDoGTuple(ρ=tupla.ρ * scale, ϕ=tupla.ϕ, σ=tupla.σ * scale)
                          for tupla in operator]
        output = self.average_response(respBank, operatorEscala, **kwargs)
        maximum_output = np.maximum(output, maximum_output)
    return maximum_output


def _Circular_Gabor__i_rotation_cosfire(self, image, operator, response_bank, **kwargs):
    maximum_output = np.zeros_like(image)
    for rotation in self.rotation_invariant:
        operatorRotacion = [CosfireCircularGaborTuple(λ=tupla.λ, θ=tupla.θ + rotation, ρ=tupla.ρ, ϕ=tupla.ϕ + rotation)
                            for tupla in operator]
        output = self.i_scale_cosfire(image, operatorRotacion, response_bank, **kwargs)
        maximum_output = np.maximum(output, maximum_output)
    return maximum_output


def _Circular_DoG__i_rotation_cosfire(self, image, operator, response_bank, **kwargs):
    maximum_output = np.zeros_like(image)
    for rotation in self.rotation_invariant:
        operatorRotacion = [CosfireCircularDoGTuple(ρ=tupla.ρ, ϕ=tupla.ϕ + rotation, σ=tupla.σ) for tupla in operator]
        output = self.i_scale_cosfire(image, operatorRotacion, response_bank, **kwargs)
        maximum_output = np.maximum(output, maximum_output)
    return maximum_output


def _Circular_Gabor__i_reflection_cosfire(self, image, operator, response_bank, **kwargs):
    maximum_output = self.i_rotation_cosfire(image, operator, response_bank, **kwargs)
    if self.reflection_invariant == 1:
        operatorI = [CosfireCircularGaborTuple(λ=tupla.λ, θ=π - tupla.θ, ρ=tupla.ρ, ϕ=π - tupla.ϕ)
                     for tupla in operator]
        output = self.i_rotation_cosfire(image, operatorI, response_bank, **kwargs)
        maximum_output = np.maximum(output, maximum_output)
    return maximum_output


def _Circular_DoG__i_reflection_cosfire(self, image, operator, response_bank, **kwargs):  # pure imagination
    return self.i_rotation_cosfire(image, operator, response_bank, **kwargs)


def _Circular_Gabor__blur_gaussian(self, bank, **kwargs):
    dic = {}
    for tupla in self._Sf_invariant:
        σ = self.alpha * tupla.ρ + self.σ0
        dic[tupla] = cv2.GaussianBlur(bank[(tupla.θ, tupla.λ)], (9, 9), σ, σ)
    return dic


def _Circular_DoG__blur_gaussian(self, bank, **kwargs):
    dic = {}
    for tupla in self._Sf_invariant:
        σ = self.alpha * tupla.ρ + self.σ0
        dic[tupla] = cv2.GaussianBlur(bank[tupla.σ], (9, 9), σ, σ)
    return dic


def _Circular_Gabor__compute_tuples(self, inputImage, **kwargs):
    # Daniel: Aqui llamar funcion que rellena parametros para invarianzas
    operator = self._Sf[:]
    if self.reflection_invariant == 1:
        operator += [CosfireCircularGaborTuple(λ=tupla.λ,
                                               θ=π - tupla.θ,
                                               ρ=tupla.ρ,
                                               ϕ=π - tupla.ϕ) for tupla in self._Sf]

    for tupla, rotation, scale in itertools.product(operator, self.rotation_invariant, self.scale_invariant):
        self._Sf_invariant.append(CosfireCircularGaborTuple(λ=tupla.λ * scale,
                                                            θ=tupla.θ + rotation,
                                                            ρ=tupla.ρ * scale,
                                                            ϕ=tupla.ϕ + rotation))
    unicos = {}
    for tupla in self._Sf_invariant:
        gabor_key = GaborKey(θ=tupla.θ, λ=tupla.λ)
        if gabor_key not in unicos:
            θ_array = np.array([gabor_key.θ])
            λ_array = np.array([gabor_key.λ])
            tt = self.compute_bank_of_responses(inputImage, θ_array=θ_array, λ_array=λ_array, **kwargs)
            unicos[gabor_key] = tt[gabor_key]
    maximum = max(value.max() for key, value in unicos.items())
    [cv2.threshold(src=image, dst=image, thresh=self.threshold_1 * maximum, maxval=maximum, type=cv2.THRESH_TOZERO)
     for key, image in unicos.items()]
    unicos = self.blur_gaussian(unicos)  ##2.1.1 Hacemos Blur
    return unicos


def _Circular_DoG__compute_tuples(self, inputImage, **kwargs):
    # Aqui llamar funcion que rellena parametros para invarianzas
    operator = self._Sf[:]
    if self.reflection_invariant == 1:
        operator += [CosfireCircularGaborTuple(λ=tupla.λ, θ=π - tupla.θ, ρ=tupla.ρ, ϕ=π - tupla.ϕ)
                     for tupla in self._Sf]

    for tupla, rotation, scale in itertools.product(operator, self.rotation_invariant, self.scale_invariant):
        new_tupla = CosfireCircularDoGTuple(ρ=tupla.ρ * scale, ϕ=tupla.ϕ + rotation, σ=tupla.σ * scale)
        self._Sf_invariant.append(new_tupla)
    unicos = {}
    for tupla in self._Sf_invariant:
        if tupla[2] not in unicos:
            l1 = np.array(1 * [tupla[2]])
            tt = self.compute_bank_of_responses(inputImage, self.filter_parameters, l1)
            unicos[tupla[2]] = tt[tupla[2]]
    maximum = max(value.max() for key, value in unicos.items())
    [cv2.threshold(src=image, dst=image, thresh=self.threshold_1 * maximum,
                   maxval=maximum, type=cv2.THRESH_TOZERO)
     for key, image in unicos.items()]
    unicos = self.blur_gaussian(unicos)  ##2.1.1 Hacemos Blur
    return unicos


def _Circular_Gabor__fit_Sf(self, **kwargs):
    operator = []
    ϕ_array = np.linspace(start=0, stop=2 * π, num=360, endpoint=False)
    for ρ in self.ρ_list:
        if ρ == 0:
            for gabor_key, response in self._prototype_bank.items():
                response_at_ρ_ϕ = response[self.center_x][self.center_y]
                if response_at_ρ_ϕ > self._maximum_response * self.threshold_2:
                    operator.append(CosfireCircularGaborTuple(λ=gabor_key.λ, θ=gabor_key.θ, ρ=0, ϕ=0))
        elif ρ > 0:
            pixel_at_ρ_ϕ_list = [Pixel(row=floor(self.center_x - ρ * sin(ϕ)), column=floor(self.center_y + ρ * cos(ϕ)))
                                 for ϕ in ϕ_array]
            maximum_responses_array = np.array([pixel_at_ρ_ϕ.maximum_response(bank=self._prototype_bank)
                                                for pixel_at_ρ_ϕ in pixel_at_ρ_ϕ_list])
            if len(np.unique(maximum_responses_array)) == 1:
                continue
            maximum_responses_array_augmented = np.zeros(maximum_responses_array.size + 1)
            maximum_responses_array_augmented[0] = maximum_responses_array[-1]
            maximum_responses_array_augmented[1:] = maximum_responses_array.copy()
            response_peaks_index = - 1 + peakutils.peak.indexes(maximum_responses_array_augmented, thres=0.2,
                                                                min_dist=22)  # 22=360/16
            for peak_index in response_peaks_index:
                for θ in self.filter_parameters.θ:
                    maximum_response = 0
                    for λ in self.filter_parameters.λ:
                        peak_pixel = pixel_at_ρ_ϕ_list[peak_index]
                        response_at_peak = self._prototype_bank[GaborKey(θ=θ, λ=λ)][peak_pixel]
                        if (response_at_peak > self.threshold_2 * self._maximum_response) and (
                                response_at_peak > maximum_response):
                            maximum_response = response_at_peak
                            λ_maximum = λ
                    if maximum_response > 0:
                        operator.append(CosfireCircularGaborTuple(λ=λ_maximum, θ=θ, ρ=ρ, ϕ=peak_index * (π / 180.0)))

    return operator


def _Circular_DoG__fit_Sf(self, **kwargs):
    operator = []
    ϕ_array = np.linspace(start=0, stop=2 * π, num=360, endpoint=False)
    for ρ in self.ρ_list:
        if ρ == 0:
            for σ in self.filter_parameters.σ:
                response_at_center = self._prototype_bank[σ][self.center_x][self.center_y]
                if response_at_center > self._maximum_response * self.threshold_2:
                    operator.append(CosfireCircularDoGTuple(ρ=0, ϕ=0, σ=σ))
        elif ρ > 0:
            listMax = np.zeros(360)
            direcciones = []
            for k, ϕ in enumerate(ϕ_array):
                yi = floor(self.center_y + ρ * cos(ϕ))
                xi = floor(self.center_x - ρ * sin(ϕ))
                response_at_center = 0
                rows, cols = self.prototype_image.shape
                if xi >= 0 and yi >= 0 and xi < rows and yi < cols:
                    for gabor_key, response in self.prototype_response_to_filters.items():
                        if response[xi][yi] > response_at_center:
                            response_at_center = response[xi][yi]
                listMax[k] = response_at_center
                direcciones.append((xi, yi))
            if len(np.unique(listMax)) == 1:
                continue
            listMax1 = np.zeros(listMax.size + 1)
            listMax1[1:] = listMax.copy()
            index = peakutils.peak.indexes(listMax1, thres=0.2, min_dist=22)  # 22=360/16
            index = list(index - 1)
            index = np.array(index)
            for k in range(index.size):
                for σ, response in self._prototype_bank.items():
                    response_at_bearing = response[direcciones[index[k]][0]][direcciones[index[k]][1]]
                    if response_at_bearing > self.threshold_2 * self._maximum_response:
                        operator.append(CosfireCircularDoGTuple(ρ=ρ, ϕ=index[k] * (π / 180.0), σ=σ))
    return operator


strategies_function_names = ['fit_Sf', 'compute_tuples', 'blur_gaussian', 'compute_bank_of_responses',
                             'i_scale_cosfire', 'i_rotation_cosfire', 'i_reflection_cosfire', ]
strategies_dictionary = {
    outer_key: {key: globals()['_' + '_'.join(outer_key.split()) + '__' + key] for key in strategies_function_names}
    for outer_key in ['Circular Gabor', 'Circular DoG']
}
