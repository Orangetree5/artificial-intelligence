from PIL import Image
from numpy import complex
import numpy as np



image = Image.new("RGB", (1000, 1000), color = (0, 0, 0))
pixels = image.load()