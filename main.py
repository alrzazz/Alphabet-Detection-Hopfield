import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont
from hopfield import hofield
import os

def image_to_np(path):
    im = Image.open(path)
    im_np = np.asarray(im)
    im_np = np.where(im_np<128, -1, 1)
    return im_np

def generate_noise(data, precentage):
    noise_indecies = np.random.randint(low=0, high=data.size, size=int(precentage*data.size))
    data[noise_indecies] *= -1
    return data

def calculate_error(data1, data2):
    data = data1 - data2
    e = np.count_nonzero(data) / data.size
    return np.round(e,4)

# test the hopfield neural network
if __name__ == "__main__":

    if not os.path.isdir("test_result"):
        os.mkdir("test_result")
    test_directory = os.path.join(os.getcwd(), "test_result")

    for font_size in [16, 32, 64]:
        font_directory = os.path.join(os.getcwd(), "test_result", str(font_size))
        if not os.path.isdir(font_directory):
            os.mkdir(font_directory)

        net = hofield(neurons=3000)

        # train hopfield
        train_directory = os.path.join(os.getcwd(), "train_data", str(font_size))
        train_data = os.listdir(train_directory)
        for img in train_data:
            data = image_to_np(os.path.join(train_directory, img))
            data = np.reshape(data, (data.size))
            net.train(data=data)

        # test hopfield
        errors = {0.1:0, 0.3:0, 0.6:0}
        for img in train_data:
            test = image_to_np(os.path.join(train_directory, img))
            x = 1
            for noise in [0.1,0.3,0.6]:
                data = np.copy(test)
                data = np.reshape(data, (data.size))
                noisy_data = generate_noise(data, noise)
                
                plt.subplot(2,3,x)
                plt.title("noise=" + str(noise))
                plt.imshow(np.reshape(noisy_data, test.shape), cmap='gray')

                res = net.predict(data=noisy_data)
                res = np.reshape(res, test.shape)
                e = calculate_error(test,res)

                plt.subplot(2,3,x+3)
                plt.title("e=" + str(e))
                plt.imshow(res, cmap='gray')

                errors[noise] += e
                x += 1
            plt.savefig(os.path.join(font_directory, img + ".png"))
        print(errors)

