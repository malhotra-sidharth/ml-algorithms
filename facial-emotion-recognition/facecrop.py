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


  def separate_classes_for_dataloader(self, images_dir, labels, image_list):
    """
    Compiles all images in class folders to be used directly by
    PyTorch Dataloader. Also resizes the images to size 285x285
    :param images_dir:
    :param labels:
    :param image_list:
    :return:
    """
    # get all unique classes
    classes = labels['emotion'].unique()
    classes = set(s.lower() for s in classes)

    # create folders
    path = images_dir + '/' + 'dataset/'
    if not os.path.exists(path):
      os.mkdir(path)

    for cl in classes:
      path = images_dir + '/' + 'dataset/' + cl
      if not os.path.exists(path):
        os.mkdir(path)

    # put images to their respective folders
    for image in image_list:
      cl = labels['emotion'][labels['image'] == image]
      if not cl.empty:
        cl = cl.values[-1]
        path = images_dir + '/' + 'dataset/' + cl
        img_path = images_dir + '/'  + image
        img = cv2.imread(img_path, 0)
        new_img = cv2.resize(img, (285, 285))
        save_path = path + '/' + image
        cv2.imwrite(save_path, new_img)