from PIL import Image
import numpy as np
import os

def flip_image(image_path, saved_location):
    """
    Flip or mirror the image
    @param image_path: The path to the image to edit
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    rotated_image = image_obj.transpose(Image.FLIP_LEFT_RIGHT)
    rotated_image.save(saved_location)


if __name__ == '__main__':
    base_dir = 'files_dependencies/gestures/images'
    train_dir = os.path.join(base_dir, 'train')

    labels_file = open('files_dependencies/gestures/labels.txt', 'r')

    CATEGORIES = []
    CATEGORIES = labels_file.read().split(',')
    class_names = np.array(CATEGORIES)

    for category in CATEGORIES:
        path = os.path.join(train_dir,category)  # create path
        countt = 0
        for image in os.listdir(path):
            countt += 1
            try:
                flip_image(os.path.join(path,image), os.path.join(path,'mirrored-'+image))
            except:
                pass
