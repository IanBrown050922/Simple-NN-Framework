import numpy as np

def __list_images(images: np.ndarray) -> list:
    image_list = []

    for i in range(images.shape[0]):
        img = images[i,:]
        img = img / 255 # normalize
        image_list.append(img)
    
    return image_list

def __list_labels(labels: np.ndarray, num_classes: int) -> list:
    label_list = []

    for i in range(labels.shape[0]):
        label_number = int(labels[i, 0])
        label = np.zeros(num_classes, dtype=np.uint)
        label[label_number] = 1
        label_list.append(label)
    
    return label_list

def process_data(images: np.ndarray, labels: np.ndarray, num_classes: int) -> list:
    image_list = __list_images(images)
    label_list = __list_labels(labels, num_classes)
    return list(zip(image_list, label_list))