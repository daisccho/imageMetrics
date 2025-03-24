import glob
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def compress_image(image_name, quality):
    image = cv2.imread(image_name)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    cv2.imwrite('1.jpeg', image, encode_param)
    compressed_image = cv2.imread('1.jpeg')
    return image.astype(np.float64), compressed_image.astype(np.float64), compressed_image


def mse(image_a, image_b):
    err = np.sum((image_a.astype(np.float64) - image_b.astype(np.float64)) ** 2)/float(image_a.shape[0] * image_b.shape[1])
    return err


def psnr(image_a, image_b):
    img1 = image_a.astype(np.float64)
    img2 = image_b.astype(np.float64)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse(img1, img2)))


def ssim(image_a, image_b):
    image_a = image_a.astype(np.float64)
    image_b = image_b.astype(np.float64)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    mu1 = cv2.GaussianBlur(image_a, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(image_b, (11, 11), 1.5)
    sigma1sq = cv2.GaussianBlur(image_a ** 2, (11, 11), 1.5) - mu1 ** 2
    sigma2sq = cv2.GaussianBlur(image_b ** 2, (11, 11), 1.5) - mu2 ** 2
    sigma12 = cv2.GaussianBlur(image_a * image_b, (11, 11), 1.5) - mu1 * mu2

    ssim_ = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1sq + sigma2sq + C2))
    return np.mean(ssim_)


if __name__ == "__main__":
    # Путь к исходному изображению
    image_path = glob.glob('stalingrad2.jpg')
    im = cv2.imread(image_path[0])
    #cv2.imshow("Image", im)

    quality_arr = []
    mse_arr = []
    psnr_arr = []
    ssim_arr = []

    for quality in tqdm(range(0, 101, 1)):
        # Сжатие изображения и сохранение результата
        orig, jpeged, image = compress_image(image_path[0], quality)
        #if (quality == 10) | (quality == 80) | (quality == 85):
        #cv2.imshow('JPEG ' + str(quality), image)

        quality_arr.append(quality)
        mse_arr.append(mse(orig, jpeged))
        psnr_arr.append(psnr(orig, jpeged))
        ssim_arr.append(ssim(orig, jpeged))

    df = pd.DataFrame({
        'Quality': quality_arr,
        'MSE': mse_arr,
        'PSNR': psnr_arr,
        'SSIM': ssim_arr
    })
    print(df)

    plt.figure(figsize=(10,8))

    plt.subplot(1, 3, 1)
    plt.plot(quality_arr, mse_arr, color='b', label='MSE')
    plt.xlabel('Quality')
    plt.ylabel('Metrics')
    plt.title('Graph of MSE')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(quality_arr, ssim_arr, color='y', label='SSIM')
    plt.xlabel('Quality')
    plt.ylabel('Metrics')
    plt.title('Graph of SSIM')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(quality_arr, psnr_arr, color='r', label='PSNR')
    plt.xlabel('Quality')
    plt.ylabel('Metrics')
    plt.title('Graph of PSNR')
    plt.legend()
    plt.grid(True)
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
