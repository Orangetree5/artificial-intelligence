from PIL import Image;
from numpy import complex;
import numpy as np;



imageWidth = 2000
imageHeight = 2000
xLength = 10
yLength = 10

steps = np.arange(-xLength/2, xLength/2, xLength/imageWidth)

image = Image.new("RGB", (imageWidth, imageHeight), color = (255, 255, 255))
pixels = image.load()


for a in range(0, imageWidth):
    pixels[a, int(imageHeight/2)] = (0, 0, 0)

for a in range(0, imageHeight):
    pixels[int(imageWidth/2), a] = (0, 0, 0)

for x in range(0, imageWidth):
    y = np.sin(steps[x])

    yReshaped = ((y * (imageHeight - 1)) / (2 * yLength))
    yFlipped = (imageHeight / 2) - yReshaped

    if(abs(yReshaped) < imageHeight/2):
        pixels[x, yFlipped] = (255, 0, 0)



image.show()