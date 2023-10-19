from django.shortcuts import render, redirect
import cv2
from .models import Image
from .forms import ImageForm
from django.conf import settings
import os
import numpy as np
import matplotlib.pyplot as plt
import pywt


def home(request):
    form = ImageForm()
    return render(request, 'home.html', {'form': form})

def result(request):
    result_path = '/mywebsite/media/image.jpg'
    return render(request, 'result.html', {'result_path': result_path})


def median_filter(image, kernel_size):
    # اعمال فیلتر Median
    median_image = cv2.medianBlur(image, kernel_size)
    
    return median_image


def image_smoothing(image, kernel_size):
    # اعمال فیلتر هموارسازی بر روی تصویر
    smoothed_image = cv2.blur(image, (kernel_size, kernel_size))

    return smoothed_image

def roberts_filter(image):
    # تبدیل تصویر به سیاه و سفید
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # اعمال فیلتر Roberts در جهت افقی
    roberts_x = cv2.filter2D(gray_image, -1, np.array([[1, 0], [0, -1]]))
    
    # اعمال فیلتر Roberts در جهت عمودی
    roberts_y = cv2.filter2D(gray_image, -1, np.array([[0, 1], [-1, 0]]))
    
    # ترکیب نتایج در جهت افقی و عمودی
    roberts_combined = cv2.addWeighted(roberts_x, 0.5, roberts_y, 0.5, 0)
    
    return roberts_combined

def wavelet_filter(image):
    # تبدیل تصویر به سیاه و سفید
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # اعمال تبدیل Wavelet
    coeffs = pywt.dwt2(gray_image, 'haar')
    cA, (cH, cV, cD) = coeffs
    
    # بازسازی تصویر از ضرایب Wavelet
    reconstructed_image = pywt.idwt2((cA, (None, None, None)), 'haar')
    
    return reconstructed_image.astype(np.uint8)


def prewitt_filter(image):
    # تبدیل تصویر به سیاه و سفید
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # اعمال فیلتر Prewitt در جهت افقی
    prewitt_x = cv2.filter2D(gray_image, -1, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
    
    # اعمال فیلتر Prewitt در جهت عمودی
    prewitt_y = cv2.filter2D(gray_image, -1, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))
    
    # ترکیب نتایج در جهت افقی و عمودی
    prewitt_combined = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)
    
    return prewitt_combined


def mean_filter(image, kernel_size):
    # ایجاد ماتریس فیلتر میانگین
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)

    # اعمال فیلتر میانگین بر روی تصویر
    result = cv2.filter2D(image, -1, kernel)

    return result


def laplacian_filter(image):
    # اعمال فیلتر لاپلاسیان
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # تبدیل مقادیر به عدد بین 0 و 255
    laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return laplacian


def gaussian_convolution(image, kernel_size, sigma):
    # ایجاد فیلتر گوسی
    kernel = cv2.getGaussianKernel(kernel_size, sigma)

    # اعمال فیلتر گوسی بر روی تصویر
    result = cv2.filter2D(image, -1, kernel)

    return result

def sobel_filter(image):
    # اعمال فیلتر سابل در راستای افقی
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)

    # اعمال فیلتر سابل در راستای عمودی
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # محاسبه مقدار مطلق گرادیان
    gradient = np.sqrt(np.square(sobelx) + np.square(sobely))

    # تبدیل مقادیر گرادیان به عدد بین 0 و 255
    gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return gradient

def low_pass_filter(image, cutoff_freq):
    # تبدیل تصویر به فضای رنگ سیاه و سفید
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # اعمال تبدیل DFT
    dft = cv2.dft(np.float32(image_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # محاسبه مرکز فضای فرکانسی
    rows, cols = image_gray.shape
    crow, ccol = rows // 2, cols // 2

    # ایجاد ماسک فیلتر پایین گذرو
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - cutoff_freq:crow + cutoff_freq, ccol - cutoff_freq:ccol + cutoff_freq, :] = 1

    # اعمال ماسک بر روی تبدیل DFT
    dft_shift_filtered = dft_shift * mask

    # بازگرداندن تبدیل DFT به حالت اصلی
    dft_filtered = np.fft.ifftshift(dft_shift_filtered)
    image_filtered = cv2.idft(dft_filtered)

    # استخراج قسمت حقیقی تصویر فیلتر شده
    image_filtered = cv2.magnitude(image_filtered[:, :, 0], image_filtered[:, :, 1])

    # نرمال‌سازی مقادیر تصویر فیلتر شده
    image_filtered = cv2.normalize(image_filtered, None, 0, 255, cv2.NORM_MINMAX)
    image_filtered = cv2.convertScaleAbs(image_filtered)

    return image_filtered


def wavelet_transform(image):
    # تبدیل تصویر به فضای رنگ سیاه و سفید
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # اعمال تبدیل ویولت
    coeffs = pywt.dwt2(image, 'haar')

    return coeffs

def dft_transform(image):
    # تبدیل تصویر به فضای رنگ سیاه و سفید
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # تبدیل DFT
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)

    # محاسبه طیف مقیاس
    magnitude_spectrum = np.log(1 + cv2.magnitude(dft[:, :, 0], dft[:, :, 1]))

    # نرمال‌سازی مقادیر طیف مقیاس
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    magnitude_spectrum = cv2.convertScaleAbs(magnitude_spectrum)

    return magnitude_spectrum


def sharpen_filter(image):
    # تعریف ماتریس فیلتر شارپن
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    # اعمال فیلتر شارپن بر روی تصویر
    result = cv2.filter2D(image, -1, kernel)

    return result

def process_image(self):
    # خواندن تصویر
    image = cv2.imread(self.cleaned_data['image'].path)

    # اعمال فیلتر شارپن
    result = sharpen_filter(image)

    return result


def process_transform(request):
    if request.method == 'POST':
        if 'fft_transform' in request.POST:
            image_path = os.path.join(settings.MEDIA_ROOT, 'image.jpg')
            image = cv2.imread(image_path)

            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.0

            f = np.fft.fft2(image_gray)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.where(np.abs(fshift) == 0, 1, np.abs(fshift)))

            plt.imshow(magnitude_spectrum, cmap='gray')
            plt.axis('off')
            plt.show()

            magnitude_spectrum_path = os.path.join(settings.MEDIA_ROOT, 'magnitude_spectrum.png')
            plt.imsave(magnitude_spectrum_path, magnitude_spectrum, cmap='gray')

            message = "عکس با موفقیت ذخیره شد."
            return render(request, 'result.html', {'message': message, 'magnitude_spectrum_path': magnitude_spectrum_path})
        elif 'sharpen_transform' in request.POST:
        
          if request.method == 'POST':
           form = ImageForm(request.POST, request.FILES)

           if form.is_valid():
            # ذخیره تصویر آپلود شده
            image_path = os.path.join(settings.MEDIA_ROOT, 'image.jpg')
            with open(image_path, 'wb') as f:
                f.write(form.cleaned_data['image'].read())

            # خواندن تصویر
            image = cv2.imread(image_path)

            # اعمال فیلتر شارپن
            result = sharpen_filter(image)

            # نمایش تصویر نتیجه
            cv2.imshow('Sharpen Filter', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            return render(request, 'result.html')
        
        elif "welwet_transform" in request.POST:
         if request.method == 'POST':
          form = ImageForm(request.POST, request.FILES)
          if form.is_valid():
             # ذخیره تصویر آپلود شده
            image_path = os.path.join(settings.MEDIA_ROOT, 'image.jpg')
            with open(image_path, 'wb') as f:
                f.write(form.cleaned_data['image'].read())
                
            # خواندن تصویر
            image = cv2.imread(image_path)

            # اعمال تبدیل ویولت
            coeffs = wavelet_transform(image)

            # نمایش ضرایب تبدیل
            cA, (cH, cV, cD) = coeffs
            cv2.imshow('Approximation (cA)', cA)
            cv2.imshow('Horizontal Detail (cH)', cH)
            cv2.imshow('Vertical Detail (cV)', cV)
            cv2.imshow('Diagonal Detail (cD)', cD)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return render(request, 'result.html')
        elif "DFT_transform" in request.POST:
           if request.method == 'POST':
             form = ImageForm(request.POST, request.FILES)
           if form.is_valid():
            # ذخیره تصویر آپلود شده
            image_path = os.path.join(settings.MEDIA_ROOT, 'image.jpg')
            with open(image_path, 'wb') as f:
                f.write(form.cleaned_data['image'].read())
            
            image = cv2.imread(image_path)
            
            # اعمال تبدیل DFT
            result = dft_transform(image)
            cv2.imshow('dft_transform', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # ذخیره تصویر نتیجه
            
            result_path = os.path.join(settings.MEDIA_ROOT, 'result.jpg')
            cv2.imwrite(result_path, result)
            # ارسال تصویر نتیجه به کاربر
            return render(request, 'result.html', {'result_path': result_path})
        elif "guassian_transform" in request.POST:
           if request.method == 'POST':
             form = ImageForm(request.POST, request.FILES)
           if form.is_valid():
            # ذخیره تصویر آپلود شده
            image_path = os.path.join(settings.MEDIA_ROOT, 'image.jpg')
            with open(image_path, 'wb') as f:
                f.write(form.cleaned_data['image'].read())
            image = cv2.imread(image_path)

            result = gaussian_convolution(image, 5, 1)
            cv2.imshow('Gaussian Convolution', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()       
            return render(request, 'result.html')
        elif "sobel_transform" in request.POST:
           if request.method == 'POST':
             form = ImageForm(request.POST, request.FILES)
           if form.is_valid():
            image_path = os.path.join(settings.MEDIA_ROOT, 'image.jpg')
            with open(image_path, 'wb') as f:
                   f.write(form.cleaned_data['image'].read())
            image = cv2.imread(image_path)
            result = sobel_filter(image)
            cv2.imshow('Sobel Filter', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return render(request, 'result.html')
        elif "Laplacian_Filter" in request.POST:
           if request.method == 'POST':
             form = ImageForm(request.POST, request.FILES)
           if form.is_valid():
            image_path = os.path.join(settings.MEDIA_ROOT, 'image.jpg')
            with open(image_path, 'wb') as f:
                   f.write(form.cleaned_data['image'].read())
            image = cv2.imread(image_path)
            result = laplacian_filter(image)
            cv2.imshow('Laplacian_Filter', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return render(request, 'result.html')
        elif "low_pass_filter" in request.POST:
           if request.method == 'POST':
             form = ImageForm(request.POST, request.FILES)
           if form.is_valid():
            image_path = os.path.join(settings.MEDIA_ROOT, 'image.jpg')
            with open(image_path, 'wb') as f:
                   f.write(form.cleaned_data['image'].read())
            image = cv2.imread(image_path)
            cutoff_freq = 30
            result = low_pass_filter(image, cutoff_freq)
            cv2.imshow('Low Pass Filter', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return render(request, 'result.html')
        elif "mean_filter" in request.POST:
           if request.method == 'POST':
             form = ImageForm(request.POST, request.FILES)
           if form.is_valid():
            image_path = os.path.join(settings.MEDIA_ROOT, 'image.jpg')
            with open(image_path, 'wb') as f:
                   f.write(form.cleaned_data['image'].read())
            image = cv2.imread(image_path)
            result = mean_filter(image, 5)
            # نمایش تصویر نرم شده
           cv2.imshow('Mean Filter', result)
           cv2.waitKey(0)
           cv2.destroyAllWindows()
           return render(request, 'result.html')
        elif "Prewitt_Filter" in request.POST:
           if request.method == 'POST':
             form = ImageForm(request.POST, request.FILES)
           if form.is_valid():
            image_path = os.path.join(settings.MEDIA_ROOT, 'image.jpg')
            with open(image_path, 'wb') as f:
                   f.write(form.cleaned_data['image'].read())
            image = cv2.imread(image_path)
            result = prewitt_filter(image)
             # اجرای تابع تبدیل
            prewitt_filter(image)  
            cv2.imshow("Prewitt Filter", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return render(request, 'result.html')
        elif "Roberts_Filter" in request.POST:
           if request.method == 'POST':
             form = ImageForm(request.POST, request.FILES)
           if form.is_valid():
            image_path = os.path.join(settings.MEDIA_ROOT, 'image.jpg')
            with open(image_path, 'wb') as f:
                   f.write(form.cleaned_data['image'].read())
            image = cv2.imread(image_path)
            resault = roberts_filter(image)
            roberts_filter(image) 
            cv2.imshow("Roberts Filter", resault )
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return render(request, 'result.html')
        elif "image_smoothing" in request.POST:
           if request.method == 'POST':
             form = ImageForm(request.POST, request.FILES)
           if form.is_valid():
            image_path = os.path.join(settings.MEDIA_ROOT, 'image.jpg')
            with open(image_path, 'wb') as f:
                   f.write(form.cleaned_data['image'].read())
            image = cv2.imread(image_path)
            kernel_size = 5
            result = image_smoothing(image, kernel_size)
            cv2.imshow('Smoothed Image', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return render(request, 'result.html')
        elif "Wavelet_Filter" in request.POST:
           if request.method == 'POST':
             form = ImageForm(request.POST, request.FILES)
           if form.is_valid():
            image_path = os.path.join(settings.MEDIA_ROOT, 'image.jpg')
            with open(image_path, 'wb') as f:
                   f.write(form.cleaned_data['image'].read())
            image = cv2.imread(image_path)
            result = wavelet_filter(image)
            cv2.imshow("Wavelet Filter", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return render(request, 'result.html')
        elif "median_filter" in request.POST:
           if request.method == 'POST':
             form = ImageForm(request.POST, request.FILES)
           if form.is_valid():
            image_path = os.path.join(settings.MEDIA_ROOT, 'image.jpg')
            with open(image_path, 'wb') as f:
                   f.write(form.cleaned_data['image'].read())
            image = cv2.imread(image_path)
            result = median_filter(image, 3)
            cv2.imshow("Median Filter", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return render(request, 'result.html')
        
