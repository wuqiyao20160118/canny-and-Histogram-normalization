import numpy as np
from math import exp, pi
import pylab
import matplotlib.pyplot as plt
from PIL import Image


def trans_gray_plt(img_dir):
    # using matplotlib.pyplot.imread()
    img = plt.imread(img_dir)      # (R, G, B) range from 0 to 255
    return np.dot(img[..., :3], [0.299, 0.587, 0.114])


def trans_gray_PIL(img_dir):
    # using PIL.Image.convert()
    img = Image.open(img_dir)
    gray_convert = img.convert('L')
    gray_convert.save('gray_lena.png')
    gray = np.asarray(gray_convert)
    return gray


def gaussian_filter(size=5):
    sigma1 = sigma2 = 1
    total = 0
    gaussian = np.zeros([size, size])
    for i in range(size):
        for j in range(size):
            gaussian[i, j] = exp(-1 / 2 * (np.square(i - int((size + 1) / 2)) / np.square(sigma1)
                            + (np.square(j - int((size + 1) / 2)) / np.square(sigma2)))) / (2 * pi * sigma1 * sigma2)
            total = total + gaussian[i, j]

    gaussian = gaussian / total
    return gaussian


def gaussian_conv(img, gauss_filter):
    width, height = gauss_filter.shape
    output_width, output_height = img.shape
    output = np.zeros((output_width - width + 1, output_height - height + 1))
    for w in range(output_width - width):
        for h in range(output_height - height):
            output[w, h] = np.sum(img[w:w+5, h:h+5] * gauss_filter)

    return output


def compute_grad(img, mode=0):
    width, height = img.shape
    if mode == 0:
        dx = np.zeros((width - 1, height - 1))
        dy = np.zeros((width - 1, height - 1))
        m = np.zeros((width - 1, height - 1))
        for w in range(width - 1):
            for h in range(height - 1):
                dx[w, h] = (img[w+1, h] - img[w, h] + img[w+1, h+1] - img[w, h+1]) / 2
                dy[w, h] = (img[w, h+1] - img[w, h] + img[w+1, h+1] - img[w+1, h]) / 2
                m[w, h] = np.sqrt(np.square(dx[w, h]) + np.square(dy[w, h]))
    elif mode == 1:
        dx = np.zeros((width - 2, height - 2))
        dy = np.zeros((width - 2, height - 2))
        m = np.zeros((width - 2, height - 2))
        for w in range(width - 2):
            for h in range(height - 2):
                dx[w, h] = (img[w+1, h] - img[w, h] + 2 * (img[w+1, h+1] - img[w, h+1]) + img[w+1, h+2] - img[w, h+2]) / 4
                dy[w, h] = (img[w, h+1] - img[w, h] + 2 * (img[w+1, h+1] - img[w+1, h]) + img[w+2, h+1] - img[w+2, h]) / 4
                m[w, h] = np.sqrt(np.square(dx[w, h]) + np.square(dy[w, h]))
    elif mode == 2:
        dx = np.zeros((width - 2, height - 2))
        dy = np.zeros((width - 2, height - 2))
        m = np.zeros((width - 2, height - 2))
        for w in range(width - 2):
            for h in range(height - 2):
                dx[w, h] = (img[w+2, h] - img[w, h] + img[w+2, h+1] - img[w, h+1] + img[w+2, h+2] - img[w, h+2]) / 3
                dy[w, h] = (img[w, h+2] - img[w, h] + img[w+1, h+2] - img[w+1, h] + img[w+2, h+2] - img[w+2, h]) / 3
                m[w, h] = np.sqrt(np.square(dx[w, h]) + np.square(dy[w, h]))
    else:
        raise(Exception, "Mode error!")

    return dx, dy, m


def NonMaxSuppress(dx, dy, m):
    width, height = m.shape
    NMS = np.copy(m)
    NMS[0, :] = NMS[width - 1, :] = NMS[:, 0] = NMS[:, height - 1] = 0
    for w in range(1, width-1):
        for h in range(1, height-1):
            if NMS[w, h] == 0:
                NMS[w, h] = 0
            else:
                grad_x = dx[w, h]
                grad_y = dy[w, h]
                distance = m[w, h]

                if np.abs(grad_y) > np.abs(grad_x):
                    weight = np.abs(grad_x) / np.abs(grad_y)
                    grad2 = m[w-1, h]  # previous row
                    grad4 = m[w+1, h]  # next row
                    if grad_x * grad_y > 0:  # gradient directions of x and y are the same
                        grad1 = m[w-1, h-1]
                        grad3 = m[w+1, h+1]
                    else:
                        grad1 = m[w-1, h+1]
                        grad3 = m[w+1, h-1]
                else:
                    weight = np.abs(grad_y) / np.abs(grad_x)
                    grad2 = m[w, h-1]
                    grad4 = m[w, h+1]
                    if grad_x * grad_y > 0:
                        grad1 = m[w+1, h+1]
                        grad3 = m[w-1, h-1]
                    else:
                        grad1 = m[w-1, h+1]
                        grad3 = m[w+1, h-1]
                gradX = weight * grad1 + (1 - weight) * grad2
                gradY = weight * grad3 + (1 - weight) * grad4
                if distance >= gradX and distance >= gradY:
                    NMS[w, h] = distance
                else:
                    NMS[w, h] = 0
    return NMS


def Double_threshold(data, tl, th):
    width, height = data.shape
    result = np.zeros((width, height))
    for w in range(1, width-1):
        for h in range(1, height-1):
            if data[w, h] < tl:
                result[w, h] = 0
            elif data[w, h] > th:
                result[w, h] = 1
            elif data[w+1, h]<th or data[w-1, h]<th or data[w, h+1]<th or data[w, h-1]<th or data[w-1, h-1]<th or data[w-1, h+1]<th or data[w+1, h+1]<th or data[w+1, h-1]<th:
                result[w, h] = 1
    return result


if __name__ == "__main__":
    gray_pic = trans_gray_PIL("./lena.png")
    gauss = gaussian_filter()
    gray_output = gaussian_conv(gray_pic, gauss)
    grad_x, grad_y, dis = compute_grad(gray_output, mode=2)
    NMS_img = NonMaxSuppress(grad_x, grad_y, dis)
    TL = 0.1 * np.max(NMS_img)
    TH = 3 * TL
    result_fig = Double_threshold(NMS_img, TL, TH)
    plt.imshow(result_fig, cmap="gray")
    pylab.show()
    #image = Image.fromarray(np.uint8(gray_output))
    #image.save('grayGuassian_lena.png')

