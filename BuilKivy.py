import cv2
import random
import numpy as np
from kivy.app import App
from kivy.uix.button import ButtonBehavior
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.checkbox import CheckBox
from kivy.core.window import Window
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.graphics import Color, RoundedRectangle
from kivy.utils import platform
from kivy.uix.textinput import TextInput
from PIL import Image as PILImage
from plyer import camera, filechooser
from kivy.uix.stacklayout import StackLayout
from On import mean_filter,median_filter, tuyentinh, gaussian_filter, derivative, thong_thap_filter, thong_cao_filter
# Set the background color for the app
Window.clearcolor = (1, 1, 1, 1)  # đặt màu nền cho cửa sổ của ứng dụng White background
Window.size = (400, 500)

class RoundedButton(ButtonBehavior, Label):
    def __init__(self, **kwargs):
        super(RoundedButton, self).__init__(**kwargs)
        # Thay đổi màu nền và màu chữ của nút
        self.background_color = (1, 0.8, 0.8, 1)  # Pink Light color
        self.color = (1, 1, 1, 1)  # Màu trắng cho chữ
        with self.canvas.before:
            Color(*self.background_color)
            self.rect = RoundedRectangle(radius=[10])
        self.bind(pos=self.update_rect, size=self.update_rect)

    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

class MobileApp(App):
    def build(self):
        self.title="Ứng dụng lọc ảnh"
        self.icon="logo.jpg"
        # Main layout
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10) # tạo layout hướng thẳng đứng (vertical)
        # Image widget
        self.img = Image(size_hint=(1, 0.8), allow_stretch=True, source="backgroud.jpg") #allow_stretch cho phép ảnh được điều chỉnh kích thước sao cho phù hợp với widget
        #Tạo một widget hình ảnh để hiển thị hình ảnh được chọn hoặc chụp với chiều ngang 100% và chiều dọc 80% màn hình
        layout.add_widget(self.img)
        # Control layout
        stack_layout = StackLayout(orientation='lr-bt', size_hint=(.9, 0.1), spacing=10, pos_hint={'center_x': 0.5, 'center_y': 0.5})
        # Choose button
        button_choose = RoundedButton(text='Chọn Ảnh',size_hint =(None, .5))
        button_choose.bind(on_press=self.choose_image)
        # Capture button
        button_capture = RoundedButton(text='Chụp Ảnh',size_hint =(None, .5))
        button_capture.bind(on_press=self.take_photo)
        # Filter button
        filter_button = RoundedButton(text='Chọn bộ lọc',size_hint =(None, .5)) #tạo button với class RoundedButton bên trên
        filter_button.bind(on_release=self.choose_filter_options)
        #liên kết sự kiện "on_release" của nút filter_button với phương thức choose_filter_options trong lớp MobileApp.
        # Save button
        save_button = RoundedButton(text='Lưu Ảnh',size_hint =(None, .5))
        save_button.bind(on_press=self.save_image)
        stack_layout.add_widget(button_choose)
        stack_layout.add_widget(filter_button)
        stack_layout.add_widget(save_button)
        stack_layout.add_widget(button_capture)
        # Add control layout for photo-related buttons to main layout
        layout.add_widget(stack_layout)
        # Set the current and original image paths
        self.display_image_path = None
        # Lưu trữ ảnh gốc
        self.original_image = None
        return layout
    
    def take_photo(self, *args):
        # Sử dụng plyer.camera để chụp ảnh và lưu với tên 'photo.jpg'
        camera.take_picture(filename='photo.jpg', on_complete=self.show_photo)

    def choose_image(self, *args):
        # # Sử dụng plyer.filechooser để mở dialog chọn ảnh
        filechooser.open_file(on_selection=self.show_photo, filters=['*.jpg', '*.png', '*.jpeg'])

    def show_photo(self, selection):
        # Kiểm tra xem selection có rỗng không trước khi truy cập vào phần tử đầu tiên
        if selection:
            # Lưu đường dẫn của ảnh mới chọn hoặc chụp
            self.current_image_path = selection[0] if isinstance(selection, list) else selection
            # Cập nhật nguồn của ImageView với ảnh được chọn hoặc chụp
            self.img.source = self.current_image_path
            self.img.reload()
            # Nếu chưa có ảnh gốc, lưu ảnh hiện tại vào ảnh gốc
            # if self.original_image_path is None:
            #     self.original_image_path = self.current_image_path
            # Lưu ảnh gốc khi hiển thị ảnh đầu tiên
            self.original_image = self.img.texture
        else:
            print("Không có ảnh được chọn.")

    def choose_filter_options(self, *args):
        # Create a popup to display filter options
        popup = Popup(title='Chọn Chế Độ Lọc', size_hint=(None, None), size=(400, 500))
        
        # Create a layout for filter options
        filter_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        # Filter options
        filter_options = ['Lọc trung bình', 'Lọc trung vị', 'Lọc gauss', 'Lọc tuyến tính','Đạo hàm ảnh','Lọc thông thấp','Lọc thông cao']
        for option in filter_options:
            btn = RoundedButton(text=option, size_hint_y=None, height=44)
            btn.bind(on_press=lambda btn, mode=option: self.apply_filter_mode(mode, popup))
            filter_layout.add_widget(btn)
        # Add filter layout to popup
        popup.content = filter_layout
        # Open the popup
        popup.open()

    def apply_filter_mode(self, mode, popup):
        popup.dismiss()  # Dismiss the popup
        if mode == 'Lọc trung bình':
            self.apply_mean_filter()
        elif mode == 'Lọc trung vị':
            self.apply_median_filter()
        elif mode == 'Lọc gauss':
            self.apply_gaussian_filter()
        elif mode == 'Lọc tuyến tính':
            self.show_filter_tuyentinh()
        elif mode == 'Đạo hàm ảnh':
            self.show_derivative_image_options()
        elif mode == 'Lọc thông thấp':
            self.show_thong_thap()
        elif mode == 'Lọc thông cao':
            self.show_thong_cao()        
    
    def apply_median_filter(self):
        if self.current_image_path is None:
            return
        # Đọc ảnh từ đường dẫn hiện tại
        image = cv2.imread(self.current_image_path, cv2.IMREAD_GRAYSCALE)
        # Áp dụng lọc trung bình
        filtered_image = median_filter(image)
        # Chuyển đổi mảng numpy của ảnh đã lọc thành một hình ảnh có thể hiển thị được bởi Kivy
        # Hiển thị ảnh đã lọc trên ImageView chính
        buffered_image = PILImage.fromarray(filtered_image)
        buffered_image.save("filtered_image.jpg")
        self.img.source = "filtered_image.jpg"
        self.img.reload()

    def apply_mean_filter(self):
        if self.current_image_path is None:
            return
        # Đọc ảnh từ đường dẫn hiện tại
        image = cv2.imread(self.current_image_path, cv2.IMREAD_GRAYSCALE)
        # Áp dụng lọc trung bình
        filtered_image = mean_filter(image)
        # Chuyển đổi mảng numpy của ảnh đã lọc thành một hình ảnh có thể hiển thị được bởi Kivy
        # Hiển thị ảnh đã lọc trên ImageView chính
        buffered_image = PILImage.fromarray(filtered_image)
        buffered_image.save("filtered_image.jpg")
        self.img.source = "filtered_image.jpg"
        self.img.reload()
        # #####
    def apply_gaussian_filter(self):
        if self.current_image_path is None:
            return
        # Đọc ảnh từ đường dẫn hiện tại
        image = cv2.imread(self.current_image_path, cv2.IMREAD_GRAYSCALE)
        # Áp dụng lọc trung bình
        filtered_image = gaussian_filter(image, 5, 2)
        # Chuyển đổi mảng numpy của ảnh đã lọc thành một hình ảnh có thể hiển thị được bởi Kivy
        # Hiển thị ảnh đã lọc trên ImageView chính
        buffered_image = PILImage.fromarray(filtered_image)
        buffered_image.save("filtered_image.jpg")
        self.img.source = "filtered_image.jpg"
        self.img.reload()

    def show_filter_tuyentinh(self):
        # Create a popup to display linear filter options
        popup = Popup(title='Chọn Chế Độ Lọc Tuyến Tính', size_hint=(None, None), size=(400, 400))
        # Create a layout for linear filter options
        filter_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        # Linear filter options
        linear_filter_options = ['Làm mờ', 'Làm sắc nét', 'Hiệu ứng chạm nổi', 'Hiệu ứng mờ do chuyển động']
        for option in linear_filter_options:
            btn = RoundedButton(text=option, size_hint_y=None, height=44)
            btn.bind(on_press=lambda btn, mode=option: self.apply_filter_tuyentinh(mode, popup))
            filter_layout.add_widget(btn)
        popup.content = filter_layout
        # Open the popup
        popup.open()  

    def  apply_filter_tuyentinh(self, mode, popup):
        popup.dismiss()  # Dismiss the popup
        # Kiểm tra xem có ảnh hiện tại trong ImageView không
        if self.current_image_path is not None:
            # Xác định số chế độ lọc dựa trên tên mode
            filter_modes = {
                'Làm mờ': 0,
                'Làm sắc nét': 1,
                'Hiệu ứng chạm nổi': 2,
                'Hiệu ứng mờ do chuyển động': 3
            }
            filter_type = filter_modes.get(mode)
            # Kiểm tra nếu chế độ lọc hợp lệ
            if filter_type is not None:
                # Đọc ảnh từ đường dẫn hiện tại
                image = cv2.imread(self.current_image_path, cv2.IMREAD_GRAYSCALE)
                # Áp dụng chế độ lọc tuyến tính từ mã nguồn
                filtered_image = tuyentinh(image, filter_type)
                # Hiển thị ảnh đã lọc trên ImageView chính
                buffered_image = PILImage.fromarray(filtered_image)
                buffered_image.save("filtered_image.jpg")
                self.img.source = "filtered_image.jpg"
                self.img.reload()
            else:
                print("Invalid filter mode selected.")
        else:
            print("Không có ảnh hiện thị.")
    def show_derivative_image_options(self):
        # Create a popup to display derivative filter options
        popup = Popup(title='Chọn Chế Độ Đạo Hàm', size_hint=(None, None), size=(400, 400))
        # Create a layout for derivative filter options
        filter_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        # Derivative filter options
        derivative_options = ['Robert', 'Prewitt', 'Sobel']
        for option in derivative_options:
            btn = RoundedButton(text=option, size_hint_y=None, height=44)
            btn.bind(on_press=lambda btn, mode=option: self.apply_derivative_image_mode(mode, popup))
            filter_layout.add_widget(btn)
        # Add derivative filter layout to popup
        popup.content = filter_layout
        # Open the popup
        popup.open()
    def apply_derivative_image_mode(self, mode, popup):
        popup.dismiss()  # Dismiss the popup
        # Kiểm tra xem có ảnh hiện tại trong ImageView không
        if self.current_image_path is not None:
            # Xác định số chế độ lọc dựa trên tên mode
            filter_modes = {
                'Robert': 0,
                'Prewitt': 1,
                'Sobel': 2,
            }
            filter_type = filter_modes.get(mode)
            # Kiểm tra nếu chế độ lọc hợp lệ
            if filter_type is not None:
                # Đọc ảnh từ đường dẫn hiện tại
                image = cv2.imread(self.current_image_path, cv2.IMREAD_GRAYSCALE)
                # Áp dụng chế độ đạo hàm từ mã nguồn
                filtered_image = derivative(image, filter_type)
                # Hiển thị ảnh đã lọc trên ImageView chính
                buffered_image = PILImage.fromarray(filtered_image)
                buffered_image.save("filtered_image.jpg")
                self.img.source = "filtered_image.jpg"
                self.img.reload()
            else:
                print("Invalid filter mode selected.")
        else:
            print("Không có ảnh hiện thị.")
    def show_thong_thap(self):
        # Create a popup to display derivative filter options
        popup = Popup(title='Chọn Chế Độ Lọc Thông Thấp', size_hint=(None, None), size=(400, 400))
        # Create a layout for derivative filter options
        filter_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        # Derivative filter options
        derivative_options = ['Ideal', 'Butterworth', 'Gaussian']
        for option in derivative_options:
            btn = RoundedButton(text=option, size_hint_y=None, height=44)
            btn.bind(on_press=lambda btn, mode=option: self.apply_thong_thap_mode(mode, popup))
            filter_layout.add_widget(btn)
        # Add derivative filter layout to popup
        popup.content = filter_layout
        # Open the popup
        popup.open()
    def apply_thong_thap_mode(self, mode, popup):
        popup.dismiss()  # Dismiss the popup
        # Kiểm tra xem có ảnh hiện tại trong ImageView không
        if self.current_image_path is not None:
            # Xác định số chế độ lọc dựa trên tên mode
            filter_modes = {
                'Ideal': 0,
                'Butterworth': 1,
                'Gaussian': 2,
            }
            filter_type = filter_modes.get(mode)
            # Kiểm tra nếu chế độ lọc hợp lệ
            if filter_type is not None:
                # Đọc ảnh từ đường dẫn hiện tại
                image = cv2.imread(self.current_image_path, cv2.IMREAD_GRAYSCALE)
                # Áp dụng chế độ đạo hàm từ mã nguồn
                filtered_image = thong_thap_filter(image, filter_type)
                # Hiển thị ảnh đã lọc trên ImageView chính
                buffered_image = PILImage.fromarray(filtered_image)
                buffered_image.save("filtered_image.jpg")
                self.img.source = "filtered_image.jpg"
                self.img.reload()
            else:
                print("Invalid filter mode selected.")
        else:
            print("Không có ảnh hiện thị.")

    def show_thong_cao(self):
        # Create a popup to display derivative filter options
        popup = Popup(title='Chọn Chế Độ Lọc Thông Cao', size_hint=(None, None), size=(400, 400))
        # Create a layout for derivative filter options
        filter_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        # Derivative filter options
        derivative_options = ['Ideal', 'Butterworth', 'Gaussian']
        for option in derivative_options:
            btn = RoundedButton(text=option, size_hint_y=None, height=44)
            btn.bind(on_press=lambda btn, mode=option: self.apply_thong_cao_mode(mode, popup))
            filter_layout.add_widget(btn)
        # Add derivative filter layout to popup
        popup.content = filter_layout
        # Open the popup
        popup.open()
    def apply_thong_cao_mode(self, mode, popup):
        popup.dismiss()  # Dismiss the popup
        # Kiểm tra xem có ảnh hiện tại trong ImageView không
        if self.current_image_path is not None:
            # Xác định số chế độ lọc dựa trên tên mode
            filter_modes = {
                'Ideal': 0,
                'Butterworth': 1,
                'Gaussian': 2,
            }
            filter_type = filter_modes.get(mode)
            # Kiểm tra nếu chế độ lọc hợp lệ
            if filter_type is not None:
                # Đọc ảnh từ đường dẫn hiện tại
                image = cv2.imread(self.current_image_path, cv2.IMREAD_GRAYSCALE)
                # Áp dụng chế độ đạo hàm từ mã nguồn
                filtered_image = thong_cao_filter(image, filter_type)
               # Hiển thị ảnh đã lọc trên ImageView chính
                buffered_image = PILImage.fromarray(filtered_image)
                buffered_image.save("filtered_image.jpg")
                self.img.source = "filtered_image.jpg"
                self.img.reload()
            else:
                print("Invalid filter mode selected.")
        else:
            print("Không có ảnh hiện thị.")
    def save_image(self, *args):
        if self.current_image_path is None:
            return
        # Xác định đường dẫn và tên file cho ảnh được lưu
        if platform == 'android':
            random_number =random.randint(1, 10000)
            strr = "/sdcard/saved_image_" + str(random_number) + ".jpg"
            # Trên Android, sử dụng đường dẫn writable để lưu ảnh
            save_path = strr  # Đường dẫn có thể được thay đổi theo nhu cầu
        else:
            # Trên các hệ điều hành khác, sử dụng đường dẫn tương đối của thư mục hiện tại
            random_number =random.randint(1, 10000)
            strr = "saved_image_" + str(random_number) + ".jpg"
            save_path = strr  # Đường dẫn có thể được thay đổi theo nhu cầu
        # Đọc ảnh từ đường dẫn hiện tại
        image = cv2.imread(self.current_image_path)
        # Lưu ảnh
        cv2.imwrite(save_path, image)
        # Hiển thị thông báo hoặc thông báo lưu thành công
        print("Ảnh đã được lưu thành công tại:", save_path)
    
class PhotoImage(Image):
    if __name__ == '__main__':
        MobileApp().run()
