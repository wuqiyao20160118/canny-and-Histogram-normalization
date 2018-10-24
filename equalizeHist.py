from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2


def equalizeHist(img_dir):
    img = Image.open(img_dir).convert('L')
    img_arr = np.asarray(img)
    img_arr_flatten = img_arr.flatten()
    hist, bins = np.histogram(img_arr_flatten, 256, [0, 256])
    cdf = hist.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] + 0.5
    cdf = cdf.astype(int)
    _, _, _ = plt.hist(cdf, bins=256, normed=0, facecolor='blue', alpha=0.5)
    plt.savefig("Histogram_2")
    result = np.interp(img_arr_flatten, bins[:-1], cdf)
    result = result.reshape(img_arr.shape)
    return result

def Hist_normalization(img_dir1, img_dir2):
    # transfer images to gray-scale map
    img1 = Image.open(img_dir1).convert('L')
    img_arr1 = np.asarray(img1)
    img_arr_flatten1 = img_arr1.flatten()
    width1, height1 = img_arr1.shape
    img2 = Image.open(img_dir2).convert('L')
    img_arr2 = np.asarray(img2)
    img_arr_flatten2 = img_arr2.flatten()
    width2, height2 = img_arr2.shape
    hist1, bins1 = np.histogram(img_arr_flatten1, 256, [0, 256])
    hist2, bins2 = np.histogram(img_arr_flatten2, 256, [0, 256])

    # normalization due to different size
    hist1 = hist1 / (width1 * height1)
    hist2 = hist2 / (width2 * height2)

    # calculate the cumulated probability
    cdf1 = hist1.cumsum()
    cdf2 = hist2.cumsum()

    # calculate the difference between two probabilities
    cdf_diff = np.zeros((256, 256))
    cdf = np.zeros(256)
    for i in range(256):
        for j in range(256):
            cdf_diff[i, j] = np.abs(cdf1[i] - cdf2[j])

    for i in range(256):
        min = cdf_diff[i, 0]
        index = 0
        for j in range(256):
            if min > cdf_diff[i, j]:
                min = cdf_diff[i, j]
                index = j
        cdf[i] = index
    result = cv2.LUT(img_arr1, cdf)
    return result


if __name__ == "__main__":
    result_image = Hist_normalization("./lena.png", "./kitten.jpg")
    print(result_image)

    image = Image.fromarray(np.uint8(result_image))
    image.save('kitten_lena.png')

