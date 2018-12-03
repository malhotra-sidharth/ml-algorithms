import cv2
import os

class CropFace:
  def __init__(self):
    self.haar_face_cascade = \
      cv2.CascadeClassifier('./data/xml/haarcascade_frontalface_default.xml')


  def crop_multiple_images(self, image_names, image_dir):
    """
    Crops each image in the given directory and list of images
    :param image_names: list of image names
    :param image_dir: path to directory with images
    :return: saves cropped_faces in new dir
    """
    save_path = image_dir + '/' + 'cropped/'
    if not os.path.exists(save_path):
      os.mkdir(save_path)

    for image in image_names:
      img = cv2.imread(image_dir + '/' + image, 0)
      faces = self.haar_face_cascade\
        .detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
      # each image has only one face
      if len(faces) >= 1:
        x, y, w, h = faces[-1]
        img = img[y:y+h, x:x+w]
        cv2.imwrite(save_path + image, img)
