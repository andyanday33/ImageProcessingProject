import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtGui
from PyQt5.QtCore import *
from skimage import data,io, filters, img_as_float, color, exposure, morphology, img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet,unwrap_phase, unsupervised_wiener, estimate_sigma,denoise_nl_means)
from skimage.transform import resize, rotate, swirl, warp



class ImageWidget(QWidget):
    #Widget üzerinde yüklü olan dosyanın file pathi ve uzantısı
    filePath = ""
    _ = ""


    def __init__(self):
        super().__init__()
        self.resize(800, 600)
        self.setWindowTitle("Andayofilter-9000")

        self.button1 = QPushButton('Upload image')
        self.button1.clicked.connect(self.get_image_file)

        self.button2 = QPushButton('Save Image')
        self.button2.clicked.connect(self.save_image_file)

        self.label1 = QLabel("Please Select a Filter To Apply:")
        self.button3 = QPushButton("Apply Selected Filter")
        self.button3.clicked.connect(self.filter_image)

        self.button4 = QPushButton("Apply Changes To Original")
        self.button4.clicked.connect(self.apply_to_original)

        self.labelImage = QLabel("Original Image Displays Here")
        self.labelImage2 = QLabel("Filtered Image Displays Here")
        self.filterList = QComboBox()

        self.button5 = QPushButton("Display and Equalize Histogram")
        self.button5.clicked.connect(self.disp_eq_hist)

        self.label2 = QLabel("X:")
        self.label3 = QLabel("Y:")
        self.inputX = QLineEdit()
        self.inputY = QLineEdit()
        self.buttonResize = QPushButton("Resize Image")
        self.buttonResize.clicked.connect(self.resize_image)

        self.label4 = QLabel("ROTATE: Angle:")
        self.inputAngle = QLineEdit()
        self.buttonRotate = QPushButton("Rotate Image")
        self.buttonRotate.clicked.connect(self.rotate_image)

        self.label5 = QLabel("SWIRL IMAGE: Coordinate X, Coordinate Y, Strength, Radius")
        self.inputCoorX = QLineEdit()
        self.inputCoorY = QLineEdit()
        self.inputStrength = QLineEdit()
        self.inputRadius = QLineEdit()
        self.buttonSwirl = QPushButton("Swirl Image")
        self.buttonSwirl.clicked.connect(self.swirl_image)

        self.buttonWarp = QPushButton("Shift image down")
        self.buttonWarp.clicked.connect(self.warp_image)

        self.label6 = QLabel("INTENSITY: In range")
        self.inputMinIn = QLineEdit()
        self.inputMaxIn = QLineEdit()
        self.label7 = QLabel("Out range")
        self.inputMinOut = QLineEdit()
        self.inputMaxOut = QLineEdit()
        self.buttonIntensity = QPushButton("Rescale Intensity")
        self.buttonIntensity.clicked.connect(self.rescale_intensity)

        self.label8 = QLabel("Select a morphological operation:")
        self.morpBox = QComboBox()
        self.buttonMorp = QPushButton("Process Morphological Operation")
        self.buttonMorp.clicked.connect(self.morphology)

        #Filter options
        self.filterList.addItem("Gaussian Filter")
        self.filterList.addItem("Unsharp Masking Filter")
        self.filterList.addItem("Hybrid Hessian Filter")
        self.filterList.addItem("TV Filter")
        self.filterList.addItem("Bilateral Filter")
        self.filterList.addItem("Wavelet Denoising Filter")
        self.filterList.addItem("Wavelet Denoising With YCbCr Colorspace")
        self.filterList.addItem("Phase Unwrapping (GRAY)")
        self.filterList.addItem("Unsupervised Wiener (GRAY)")
        self.filterList.addItem("Non-Local Means")


        #Morphology options
        self.morpBox.addItem("Erosion")
        self.morpBox.addItem("Dilation")
        self.morpBox.addItem("Opening")
        self.morpBox.addItem("Closing")
        self.morpBox.addItem("White Tophat")
        self.morpBox.addItem("Black Tophat")
        self.morpBox.addItem("Skeletonize")
        self.morpBox.addItem("Convex Hull")

        layout = QGridLayout()

        layout.addWidget(self.button1, 0,0)

        layout.addWidget(self.labelImage, 1,0)
        layout.addWidget(self.labelImage2, 1,1)
        layout.addWidget(self.button2, 20,1)
        layout.addWidget(self.button4, 20,0)
        layout.addWidget(self.label1, 2,0)
        layout.addWidget(self.button3, 4,0)
        layout.addWidget(self.filterList, 3,0)
        layout.addWidget(self.button5, 5, 0)
        layout.addWidget(self.label2, 6,0)
        layout.addWidget(self.label3, 6,2)
        layout.addWidget(self.inputX, 6,1)
        layout.addWidget(self.inputY, 6,3)
        layout.addWidget(self.buttonResize, 6,4)
        layout.addWidget(self.label4, 7,0)
        layout.addWidget(self.inputAngle, 7,1)
        layout.addWidget(self.buttonRotate, 7, 2)
        layout.addWidget(self.label5, 8,0)
        layout.addWidget(self.inputCoorX, 8,1)
        layout.addWidget(self.inputCoorY,8,2)
        layout.addWidget(self.inputStrength, 8,3)
        layout.addWidget(self.inputRadius, 8,4)
        layout.addWidget(self.buttonSwirl, 8,5)
        layout.addWidget(self.buttonWarp, 9,0)
        layout.addWidget(self.label6, 10,0)
        layout.addWidget(self.inputMinIn, 10,1)
        layout.addWidget(self.inputMaxIn, 10,2)
        layout.addWidget(self.label7, 10,3)
        layout.addWidget(self.inputMinOut, 10,4)
        layout.addWidget(self.inputMaxOut, 10,5)
        layout.addWidget(self.buttonIntensity, 10,6)
        layout.addWidget(self.label8, 11,0)
        layout.addWidget(self.morpBox, 11,1)
        layout.addWidget(self.buttonMorp,11,2)


        self.setLayout(layout)

    def get_image_file(self):
        self.filePath, self._ = QFileDialog.getOpenFileName(self, 'Open Image File', 'c:\\',
                                                   "Image files (*.jpg *.jpeg *.gif)")
        self.labelImage.setPixmap(QPixmap(self.filePath).scaled(600,600))


    def save_image_file(self):
        self.filePath, self._ = QFileDialog.getSaveFileName(self, "Save Image", "", ".jpg")
        print(self.filePath)
        print(self._)
        if self.filePath == "":
            return

        p = self.labelImage2.pixmap()
        p.save(self.filePath + self._)

    def apply_to_original(self):
        self.labelImage.setPixmap(self.labelImage2.pixmap())
        img = io.imread('current.jpg')
        io.imsave('currentOrg.jpg', img)
        self.filePath = 'currentOrg.jpg'

    def filter_image(self):

        picture = io.imread(self.filePath)
        picture_filtered = picture

        if self.filterList.currentText() == "Gaussian Filter":
            picture_filtered = filters.gaussian(picture, sigma = 2)
            print("Gaussian Filter applied")
        elif self.filterList.currentText() == "Unsharp Masking Filter":
            picture_filtered = filters.unsharp_mask(picture)
            print("Unsharp Masking applied")
        elif self.filterList.currentText() == "Hybrid Hessian Filter":
            picture_filtered = filters.hessian(picture, mode ='constant')
            print("Hessian Filter applied")
        elif self.filterList.currentText() == "TV Filter":
            picture_filtered = denoise_tv_chambolle(picture, weight=0.1,multichannel=True)
            print("TV Filter applied")
        elif self.filterList.currentText() == "Bilateral Filter":
            picture_filtered = denoise_bilateral(picture, sigma_color= 0.1, sigma_spatial=15, multichannel=True)
            print("Bilateral Filter applied")
        elif self.filterList.currentText() == "Wavelet Denoising Filter":
            picture_filtered = denoise_wavelet(picture, multichannel=True, rescale_sigma=True)
            print("Wavelet Denoising Applied")
        elif self.filterList.currentText() == "Phase Unwrapping (GRAY)":
            pic = color.rgb2gray(img_as_float(picture))
            pic = exposure.rescale_intensity(pic, out_range=(0, 4 * np.pi))
            pic_wrapped = np.angle(np.exp(1j * pic))
            pic_unwrapped = unwrap_phase(pic_wrapped)
            ##fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
            ##ax1, ax2, ax3, ax4 = ax.ravel()

            ##fig.colorbar(ax1.imshow(pic, cmap='gray', vmin=0, vmax=4 * np.pi), ax=ax1)
            ##ax1.set_title('Original')

            ##fig.colorbar(ax2.imshow(pic_wrapped,cmap='gray', vmin=-np.pi, vmax=np.pi),
              ##           ax=ax2)
            ##ax2.set_title('Wrapped phase')

            ##fig.colorbar(ax3.imshow(pic_unwrapped, cmap='gray'), ax=ax3)

            ##ax3.set_title('After phase unwrapping')
            picture_filtered = pic_unwrapped
            plt.show()

            print("Phase Unwrapped applied")
        elif self.filterList.currentText() == "Unsupervised Wiener (GRAY)":
            pic = color.rgb2gray(picture)
            from scipy.signal import convolve2d as conv2
            psf = np.ones((5, 5)) / 25
            astro = conv2(pic, psf, 'same')
            astro += 0.1 * astro.std() * np.random.standard_normal(astro.shape)

            deconvolved, _ = unsupervised_wiener(pic, psf)

           ## fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5),
                                ##  sharex=True, sharey=True)

##            plt.gray()

  ##          ax[0].imshow(pic, vmin=deconvolved.min(), vmax=deconvolved.max())
    ##        ax[0].axis('off')
      ##      ax[0].set_title('Data')

        ##    ax[1].imshow(deconvolved)
          ##  ax[1].axis('off')
            ##ax[1].set_title('Self tuned restoration')

            ##fig.tight_layout()
            picture_filtered = pic
            plt.show()
        elif self.filterList.currentText() == "Non-Local Means":
            pic = img_as_float(picture)
            sigma = 0.08
            sigma_est = np.mean(estimate_sigma(pic, multichannel=True))
            print(f"estimated noise standard deviation = {sigma_est}")

            patch_kw = dict(patch_size=5,  # 5x5 patches
                            patch_distance=6,  # 13x13 search area
                            multichannel=True)
            denoise = denoise_nl_means(pic, h=0.8 * sigma_est, fast_mode=True,
                                            **patch_kw)
            picture_filtered = denoise
        elif self.filterList.currentText() == "Wavelet Denoising With YCbCr Colorspace":
            pic = img_as_float(picture)
            picture_filtered = denoise_wavelet(pic, multichannel=True, convert2ycbcr=True, rescale_sigma=True)


        io.imsave('current.jpg', picture_filtered)
        self.labelImage2.setPixmap(QPixmap('current.jpg'))
        #plt.figure(figsize=(16,4))
        #plt.subplot(141)
        #plt.imshow(picture)
        #plt.subplot(142)
        #plt.imshow(picture_filtered)
        #plt.show()

    def disp_eq_hist(self):

        from skimage.util.dtype import dtype_range
        from skimage.util import img_as_ubyte
        from skimage.filters import rank
        from skimage.morphology import disk

        def plot_img_and_hist(image, axes, bins=256):

            ax_img, ax_hist = axes
            ax_cdf = ax_hist.twinx()

            # Display image
            ax_img.imshow(image, cmap=plt.cm.gray)
            ax_img.set_axis_off()

            # Display histogram
            ax_hist.hist(image.ravel(), bins=bins)
            ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
            ax_hist.set_xlabel('Pixel intensity')

            xmin, xmax = dtype_range[image.dtype.type]
            ax_hist.set_xlim(xmin, xmax)

            # Display cumulative distribution
            img_cdf, bins = exposure.cumulative_distribution(image, bins)
            ax_cdf.plot(bins, img_cdf, 'r')

            return ax_img, ax_hist, ax_cdf

        img = img_as_ubyte(color.rgb2gray(io.imread(self.filePath)))

        img_rescale = exposure.equalize_hist(img)


        # Display results
        fig = plt.figure(figsize=(8, 5))
        axes = np.zeros((2, 2), dtype=np.object)
        axes[0, 0] = plt.subplot(2, 3, 1)
        axes[0, 1] = plt.subplot(2, 3, 2, sharex=axes[0, 0], sharey=axes[0, 0])

        axes[1, 0] = plt.subplot(2, 3, 4)
        axes[1, 1] = plt.subplot(2, 3, 5)



        ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
        ax_img.set_title('Low contrast image')
        ax_hist.set_ylabel('Number of pixels')

        ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_rescale, axes[:, 1])
        ax_img.set_title('Global equalise')


        # prevent overlap of y-axis labels
        fig.tight_layout()

        plt.show()

        io.imsave('current.jpg', img_rescale)
        self.labelImage2.setPixmap(QPixmap('current.jpg'))

    def resize_image(self):
        image = io.imread(self.filePath)
        image = resize(image, (int(self.inputX.text()), int(self.inputY.text())))
        io.imsave('current.jpg', image)
        self.labelImage2.setPixmap(QPixmap('current.jpg'))

    def rotate_image(self):
        image = io.imread(self.filePath)
        image = rotate(image, int(self.inputAngle.text()))
        io.imsave('current.jpg', image)
        self.labelImage2.setPixmap(QPixmap('current.jpg'))

    def swirl_image(self):
        image = io.imread(self.filePath)
        image = swirl(image, (int(self.inputCoorX.text()), int(self.inputCoorY.text())),
                      strength= int(self.inputStrength.text()), radius = int(self.inputRadius.text()))
        io.imsave('current.jpg', image)
        self.labelImage2.setPixmap(QPixmap('current.jpg'))

    def warp_image(self):
        image = io.imread(self.filePath)

        def shift_down(xy):
            xy[:, 1] -= 10
            return xy
        image = warp(image, shift_down)
        io.imsave('current.jpg', image)
        self.labelImage2.setPixmap(QPixmap('current.jpg'))

    def rescale_intensity(self):
        image = io.imread(self.filePath)
        image = img_as_float(image)
        image = exposure.rescale_intensity(image, in_range= (int(self.inputMinIn.text()),
                                                             int(self.inputMaxIn.text())),
                                           out_range= (int(self.inputMinOut.text()),
                                                       int(self.inputMaxOut.text())))
        io.imsave('current.jpg',image)
        self.labelImage2.setPixmap(QPixmap('current.jpg'))

    def morphology(self):
        image = img_as_ubyte(color.rgb2gray(io.imread(self.filePath)))

        selem = morphology.disk(3)

        if self.morpBox.currentText() == "Erotion":
            image = morphology.erosion(image,selem)
        elif self.morpBox.currentText() == "Dilation":
            image = morphology.dilation(image, selem)
        elif self.morpBox.currentText() == "Opening":
            image = morphology.opening(image, selem)
        elif self.morpBox.currentText() == "Closing":
            image = morphology.closing(image, selem)
        elif self.morpBox.currentText() == "White Tophat":
            image = morphology.white_tophat(image, selem)
        elif self.morpBox.currentText() == "Black Tophat":
            image = morphology.black_tophat(image, selem)
        elif self.morpBox.currentText() == "Skeletonize":
            image = morphology.skeletonize(image == 0)
        elif self.morpBox.currentText() == "Convex Hull":
            image = morphology.convex_hull_image(image == 0)

        io.imsave('current.jpg', image)
        self.labelImage2.setPixmap(QPixmap('current.jpg'))

if __name__ == '__main__':
    app = QApplication(sys.argv)

    demo = ImageWidget()
    demo.show()

    sys.exit(app.exec_())
