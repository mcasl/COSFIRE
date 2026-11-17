"""
============================================================
 COSFIRE Circular Gabor strategy.
============================================================

Let's create a demo shape with a couple of rectangles.
"""
import numpy as np
import cv2
from cosfire.base import (Cosfire,
                          GaborParameters,
                          π,
                          )
from matplotlib import pyplot as plt

pattern = np.zeros((256, 256))
cv2.rectangle(img=pattern, pt1=(50, 100), pt2=(100, 106), color=255, thickness=-1)
cv2.rectangle(img=pattern, pt1=(100, 100), pt2=(97, 50), color=255, thickness=-1)
plt.imshow(pattern)
plt.show()
##############################################################################
# Now let's fit a COSFIRE filter using the demo pattern as prototype:
##############################################################################

some_cosfire = Cosfire(strategy_name='Circular Gabor',
                               center_x=100,
                               center_y=100,
                               rho_list=range(0, 100, 10),
                               t1=0.99,
                               t2=0.75,
                               t3=0.9,
                               filter_parameters=GaborParameters(ksize=(10, 10), σ=5,
                                                                 θ=np.linspace(start=0, stop=π, num=30, endpoint=False),
                                                                 λ=np.linspace(start=7, stop=8, num=10, endpoint=False),
                                                                 γ=0.5, ψ=π, ktype=cv2.CV_32F),
                               sigma0=0.67,
                               alpha=0.04,
                               reflection_invariant=True,
                               scale_invariant=[0.5, 1, 2],
                               rotation_invariant=[0, 0.5*π, 1.5*π]
                               )

some_cosfire.fit(pattern)
for parameters in some_cosfire._Sf:
    print(parameters)


##############################################################################
# Once fitted let's apply the COSFIRE filter, for example to the same image.
##############################################################################
pattern_response = some_cosfire.transform(pattern)
plt.imshow(pattern_response)
plt.show()


##############################################################################
# COSFIRE was capable of identifying the response.
##############################################################################
keypoints = np.argwhere(pattern_response>0)
print(keypoints)

##############################################################################
# Let's create a different image
##############################################################################

test = np.zeros((256, 256))
cv2.rectangle(img=test, pt1=(10, 153), pt2=(60, 150), color=255, thickness=-1)
cv2.rectangle(img=test, pt1=(60, 150), pt2=(57, 100), color=255, thickness=-1)

cv2.rectangle(img=test, pt1=(180, 197), pt2=(240, 200), color=255, thickness=-1)
cv2.rectangle(img=test, pt1=(240, 200), pt2=(237, 150), color=255, thickness=-1)

cv2.rectangle(img=test, pt1=(120, 97), pt2=(160, 100), color=255, thickness=-1)
cv2.rectangle(img=test, pt1=(160, 100), pt2=(157, 150), color=255, thickness=-1)
plt.imshow(test)
plt.show()

##############################################################################
# And let's apply COSFIRE to this new image
##############################################################################
test_response = some_cosfire.transform(test)
plt.imshow(test_response)
plt.show()
test_keypoints = np.argwhere(test_response>0)
print(test_keypoints)