from PIL import Image, ImageFont
import os

def generate_train_data():
    if not os.path.isdir("train_data"):
        os.mkdir("train_data")
    for font_size in [16, 32, 64]:
        directory = os.path.join(os.getcwd(), "train_data", str(font_size))
        if not os.path.isdir(directory):
            os.mkdir(directory)
        font = ImageFont.truetype("tahoma.ttf", font_size)
        for char in "ABCDEFGHIJ":
            im = Image.Image()._new(font.getmask(char))
            path = os.path.join(directory,  char + ".bmp")
            im.save(path)

if __name__ == '__main__':
    generate_train_data()