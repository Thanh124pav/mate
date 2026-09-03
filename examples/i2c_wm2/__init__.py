"""Example of I2C agents for the Multi-Agent Tracking Environment."""

from examples.i2c_wm2 import camera, target
from examples.i2c_wm2.camera import I2CWM2CameraAgent
from examples.i2c_wm2.target import I2CWM2TargetAgent


CameraAgent = I2CWM2CameraAgent
TargetAgent = I2CWM2TargetAgent
