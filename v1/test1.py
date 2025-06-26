import numpy as np

File_oprn = open('/home/kunal/ML/1/WhatsApp Image 2025-05-18 at 15.58.58_a227afba.jpg', 'rb')

image_pixcels = File_oprn.read()
image_pixcels = np.frombuffer(image_pixcels, dtype=np.uint8)
image_size = 512*512
image_pixcels_matrix = np.empty((1,image_size))
image_pixcels_matrix = image_pixcels.reshape(1,image_size)

print(image_pixcels)
