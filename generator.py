from PIL import Image, ImageFont

def generate_train_data():
    for font_size in [16, 32, 64]:
        font = ImageFont.truetype("tahoma.ttf", font_size)
        for char in "ABCDEFGHIJ":
            im = Image.Image()._new(font.getmask(char))
            im.save("./train_data/" + str(font_size) + "/" + char + ".bmp")

if __name__ == '__main__':
    generate_train_data()
    