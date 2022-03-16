#https://duckduckgo.com/?q=how+to+convert+an+image+into+cartoon+with+python&atb=v314-6&ia=web
#https://www.askpython.com/python/examples/images-into-cartoons
#https://www.section.io/engineering-education/how-to-turn-your-photo-into-a-cartoon-using-python/
#https://towardsdatascience.com/turn-photos-into-cartoons-using-python-bb1a9f578a7e


import cv2
import numpy as np
import matplotlib.pyplot as plt

def color_quantization(img, k):
  # Transform the image
  data = np.float32(img).reshape((-1, 3))

  # Determine criteria
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

  # Implementing K-Means
  ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
  center = np.uint8(center)
  result = center[label.flatten()]
  result = result.reshape(img.shape)
  return result

img = cv2.imread("IMG-20210107-WA0019.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10,10))
plt.imshow(img)
plt.axis("off")
plt.title("Original Image")
plt.show()



gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)
plt.figure(figsize=(10,10))
plt.imshow(gray,cmap="gray")
plt.axis("off")
plt.title("Grayscale Image")
plt.show()

edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
plt.figure(figsize=(10,10))
plt.imshow(edges,cmap="gray")
plt.axis("off")
plt.title("Edged Image")
plt.show()





totalColour = 12
img = color_quantization(img, totalColour)

plt.figure(figsize=(10,10))
plt.imshow(img)
plt.axis("off")
plt.title("quantization Image")
plt.show()



color = cv2.bilateralFilter(img, 9, 250, 250)
cartoon = cv2.bitwise_and(color, color, mask=edges)
plt.figure(figsize=(10,10))
plt.imshow(cartoon,cmap="gray")
plt.axis("off")
plt.title("Cartoon Image")
plt.show()