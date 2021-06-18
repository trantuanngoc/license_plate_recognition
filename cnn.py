import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model


class CNN:
    def __init__(self):
        self.model = load_model("./CNN/weight.h5")

    def get_char_name(self, image):
        DICT = {0: "1",1: "2",2: "3",4: "4",5: "5",6: "6",7: "7",8: "8",9: "9",10: "A",11: "B",
            12: "C",13: "D",14: "E",15: "F",16: "G",17: "H",18: "I",19: "J",20: "K",21: "L",
            22: "M",23: "N",24: "Q",25: "R",26: "S",27: "T",28: "U",29: "V",30: "W",31: "X",
            32: "Y",33: "Z"}
        image = tf.expand_dims(image, 0)
        predictions = self.model.predict(image)
        return DICT[np.argmax(predictions[0])]
