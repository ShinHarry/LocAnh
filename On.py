import cv2
import numpy as np
def convolution_same(image, kernel):
     # Lấy kích thước kernel(nhân)
    kernel_height, kernel_width = kernel.shape[:2] # Lấy kênh chiều cao và rộng của kernel

    # Padding ảnh với giá trị zero
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    # Thêm padding

    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # Thực hiện nhân chập
    result = cv2.filter2D(padded_image, -1, kernel)

    # Cắt phần dư ra khi padding
    result = result[pad_height:image.shape[0]+pad_height, pad_width:image.shape[1]+pad_width]

    return result
#Lọc tuyến tính
def tuyentinh(image, number):
    #Làm mờ ảnh
    if(number ==0):
        kernel =0.2* np.array([[0,0,0,0,0],
                           [0,1/9,1/9,1/9, 0],
                           [0,1/9,1/9,1/9, 0],
                           [0,1/9,1/9,1/9,0],
                           [0,0,0,0,0]])
        image_tuyentinh = convolution_same(image, kernel)
        return image_tuyentinh
    #Làm sắc nét ảnh
    if(number ==1):
        kernel = 0.2*np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        image_tuyentinh = convolution_same(image, kernel)
        return image_tuyentinh
    #Hiệu ứng chạm nổi
    if(number ==2):
        kernel =0.2* np.array([[-1,-1,-1,-1,0],
                            [-1,-1,-1,0, 1],
                            [-1,-1,0,1, 1],
                            [-1,0,1,0,1],
                            [0,1,1,1,1]])
        image_tuyentinh = convolution_same(image, kernel)
        return image_tuyentinh
    if(number ==3):
        kernel = 0.2*np.array([[1,0,0,0,0],
                            [0,1,0,0,0],
                            [0,0,1,0,0],
                            [0,0,0,1,0],
                            [0,0,0,0,1]])
        return image_tuyentinh
#Lọc trung bình
def mean_filter(image):
    kernel= np.array([[1/9,1/9,1/9],
                      [1/9,1/9,1/9],
                      [1/9,1/9,1/9]])
    mean_filtered =convolution_same(image, kernel)
    # cv2.imshow("Mean Filter", mean_filtered)
    return mean_filtered

def median_filter(image, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    if kernel_size % 2 == 0:
        kernel_size += 1
    convoluted_image = convolution_same(image, kernel)
    median_filtered = cv2.medianBlur(convoluted_image, kernel_size)
    return median_filtered

#Lọc Gausian
def gaussian_filter(image, kernel_size, sigma):
    # Lấy kích thước ảnh
    height, width = image.shape

    # Tạo kernel Gaussian
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i, j] = np.exp(-((i - center)**2 + (j - center)**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)  # Chuẩn hóa kernel

    # Thêm viền cho ảnh
    padded_image = np.pad(image, ((center, center), (center, center)), mode='constant')

    # Áp dụng lọc Gaussian
    result = cv2.filter2D(padded_image, -1, kernel)
    # Cắt phần padding
    result = result[center:image.shape[0]+center, center:image.shape[1]+center]
    return result

#Đạo hàm ảnh
def derivative(image, number):
    # Nhân của X,Y
    if(number == 0):
        kernel_x = np.array([[1,0],
                             [0,-1]])
        kernel_y = np.array([[0,1],
                             [-1,0]])
    if(number == 1):
        kernel_x = np.array([[-1, 0, 1],
                             [-1, 0, 1],
                             [-1, 0, 1]])
        kernel_y = np.array([[-1, -1, -1],
                             [0, 0, 0],
                             [1, 1, 1]])
    if(number ==2 ):
        kernel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
        kernel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])
    #Lọc nhiễu bằng gaussian_filter
    img_gau_filtered = gaussian_filter(image, kernel_size=5, sigma=2)
    # Lấy kênh chiều cao và rộng của kernel
    kernel_height, kernel_width = kernel_x.shape[:2]
    # Padding ảnh với giá trị zero
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    # Bù ảnh
    padded_image = np.pad(img_gau_filtered, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    
    # Đạo hàm X,Y
    derivative_x = cv2.filter2D(padded_image, -1, kernel_x)
    derivative_y = cv2.filter2D(padded_image, -1, kernel_y)
    #Trị tuyệt đối X,Y
    abs_grad_x = cv2.convertScaleAbs(derivative_x)
    abs_grad_y = cv2.convertScaleAbs(derivative_y)
    # Kết hợp trị tuyệt đối của đạo hàm X,Y 
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad
def post_process_image(image):
    final = np.uint8(cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX))
    return final


def ideal_low_pass_filter(image, cutoff):
    d0=cutoff
    rows, columns = image.shape[:2]
    mask = np.zeros((rows, columns), dtype=np.uint8)
    mid_R, mid_C = rows//2, columns//2
    thickness=-1
    mask = cv2.circle(mask, (mid_C, mid_R), d0 , 255, thickness)
    # mask = cv2.circle(mask, (mid_C, mid_R), d0 , 0, thickness)
    return mask

def butterworth_low_pass_filter(image, cutoff, degree):
    rows, columns = image.shape[:2]
    cutoff2 = cutoff**2
    Y, X = np.ogrid[:rows, :columns]
    center_y, center_x = rows/2, columns/2
    distance = (X - center_x)**2 +(Y-center_y)**2
    butterworth_filter = 1/(1+(distance/cutoff2)**degree)
    return butterworth_filter

def gaussian_low_pass_filter(image, cutoff):
    #heigth = rows, width = columns
    rows, columns = image.shape[:2]
    Y, X = np.ogrid[:rows, :columns]
    center_y, center_x = rows/2, columns/2
    # X,Y = np.ogrid[:rows, :columns]
    # center_x, center_y = rows/2, columns/2
    cutoff2 = cutoff**2
    distance = (X - center_x)**2 +(Y-center_y)**2
    gaussian_filter=np.exp(-distance/(2*cutoff2))
    return gaussian_filter

def thong_thap_filter(image, number):
    # Các bước lọc ảnh
    # 1.Chuyển ảnh sang miền tần số bằng biến đổi fourier
    fft= np.fft.fft2(image)
    # 2.Chuyển fft sang giữ các miền tần số thấp
    shift_fft=np.fft.fftshift(fft)
    # 3.Lấy mặt nạ
    if(number==0):
        mask=ideal_low_pass_filter(image, cutoff=10)
    if(number==1):
        mask=butterworth_low_pass_filter(image, cutoff=10, degree=2)
    if(number==2):
        mask=gaussian_low_pass_filter(image, cutoff=10)
    # mask=butterworth_low_pass_filter(image,cutoff=50, degree=1)
    # mask=gaussian_low_pass_filter(image, cutoff=20)
    # 4.filter the image frequency based on the mask(Convolution theorem)
    filtered_image=np.multiply(mask, shift_fft)
    # 5.Compute the inverse shift
    shift_ifft = np.fft.ifftshift(filtered_image)
    # 6.Compute the inverse fourier transform
    ifft = np.fft.ifft2(shift_ifft)
    # 7.Compute the magnitude
    mag=np.abs(ifft)
    # 8.Post precessing
    filtered_image = post_process_image(mag)
    return filtered_image
def thong_cao_filter(image, number):
    # Các bước lọc ảnh
    # 1.Chuyển ảnh sang miền tần số bằng biến đổi fourier
    fft= np.fft.fft2(image)
    # 2.Chuyển fft sang giữ các miền tần số thấp
    shift_fft=np.fft.fftshift(fft)
    # 3.Lấy mặt nạ
    if(number==0):
        mask=1-ideal_low_pass_filter(image, cutoff=10)
    if(number==1):
        mask=1-butterworth_low_pass_filter(image, cutoff=10, degree=2)
    if(number==2):
        mask=1-gaussian_low_pass_filter(image, cutoff=10)
    # mask=butterworth_low_pass_filter(image,cutoff=50, degree=1)
    # mask=gaussian_low_pass_filter(image, cutoff=20)
    # 4.filter the image frequency based on the mask(Convolution theorem)
    filtered_image=np.multiply(mask, shift_fft)
    # 5.Compute the inverse shift
    shift_ifft = np.fft.ifftshift(filtered_image)
    # 6.Compute the inverse fourier transform
    ifft = np.fft.ifft2(shift_ifft)
    # 7.Compute the magnitude
    mag=np.abs(ifft)
    # 8.Post precessing
    filtered_image = post_process_image(mag)
    return filtered_image