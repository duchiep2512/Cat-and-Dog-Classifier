## Giới thiệu

Đây là dự án nhận diện ảnh mèo và chó sử dụng Deep Learning với TensorFlow và giao diện web Streamlit. Bài toán này là **phân loại ảnh nhị phân**: xác định một ảnh bất kỳ là mèo hay chó. Đây là một bài toán kinh điển trong lĩnh vực Computer Vision, thường dùng để thực hành các kỹ thuật tiền xử lý, xây dựng và huấn luyện mô hình CNN, cũng như triển khai ứng dụng AI thực tế.

## Demo Online

Trải nghiệm ứng dụng trực tiếp tại:  
[https://cat-and-dog-classifier.onrender.com/](https://cat-and-dog-classifier.onrender.com/)

## Dữ liệu

- **Nguồn dữ liệu:** Sử dụng bộ dữ liệu `cats_vs_dogs` từ [TensorFlow Datasets](https://www.tensorflow.org/datasets/community_catalog/huggingface/cats_vs_dogs).
- **Đặc điểm:** Gồm hàng ngàn ảnh mèo và chó, đa dạng về góc chụp, màu sắc, kích thước.
- **Tiền xử lý:** Ảnh được resize về kích thước 224x224, chuyển sang RGB, chuẩn hóa pixel về [0, 1]. Áp dụng các kỹ thuật tăng cường dữ liệu như lật ngang, xoay, thay đổi độ sáng và độ tương phản để tăng độ đa dạng và giảm overfitting.

## Cấu trúc thư mục & Tác dụng từng file

```
.
├── app.py                  # Ứng dụng web Streamlit để dự đoán mèo/chó từ ảnh
├── Build_model.ipynb       # Notebook huấn luyện, fine-tune và đánh giá mô hình
├── fine_tuned_best.h5      # File lưu mô hình đã huấn luyện tốt nhất (sinh ra từ notebook)
├── requirements.txt        # Danh sách các thư viện Python cần thiết
├── render.yaml             # Cấu hình deploy ứng dụng trên nền tảng Render
├── .gitattributes          # Cấu hình Git LFS cho file mô hình lớn (.h5)
```

### Giải thích chi tiết:

- **app.py:**  
  - Giao diện web cho phép người dùng tải ảnh lên hoặc nhập URL ảnh.
  - Tự động tải mô hình đã huấn luyện (`fine_tuned_best.h5`), tiền xử lý ảnh, dự đoán và hiển thị kết quả (Cat/Dog, độ tin cậy, biểu tượng).
  - Có các tab riêng cho upload file và nhập URL, hiển thị tiến trình và confidence bar.

- **Build_model.ipynb:**  
  - Notebook chi tiết toàn bộ quy trình: tải dữ liệu, tiền xử lý, tăng cường dữ liệu, xây dựng kiến trúc CNN, huấn luyện, fine-tune, lưu mô hình, đánh giá và dự đoán thử nghiệm.
  - Có các đoạn code trực quan hóa dữ liệu, biểu đồ loss/accuracy, và ví dụ dự đoán trên ảnh mới.

- **fine_tuned_best.h5:**  
  - File lưu mô hình đã huấn luyện tốt nhất, dùng để dự đoán trong app.py.
  - Được sinh ra từ quá trình huấn luyện và fine-tune trong notebook.

- **requirements.txt:**  
  - Liệt kê các thư viện cần thiết: `streamlit`, `tensorflow`, `pillow`, `numpy`, `requests`.

- **render.yaml:**  
  - Cấu hình để deploy ứng dụng trên Render (dịch vụ cloud), chỉ định build/start command, biến môi trường, runtime Python.

- **.gitattributes:**  
  - Cấu hình Git LFS để quản lý file mô hình `.h5` lớn, tránh lỗi khi push lên GitHub.

## Thuật toán & Kiến trúc mô hình

- **Tiền xử lý:**  
  - Resize ảnh về 224x224, chuyển sang RGB, chuẩn hóa pixel về [0, 1].
  - Tăng cường dữ liệu: lật ngang, xoay, thay đổi độ sáng/độ tương phản.

- **Kiến trúc CNN:**  
  - 4 lớp Conv2D (32, 64, 128, 128 filters), activation `relu`.
  - 4 lớp MaxPooling2D để giảm chiều không gian.
  - 2 lớp Dropout (0.5) để giảm overfitting.
  - 1 lớp Flatten, 1 lớp Dense 512, 1 lớp Dense đầu ra (sigmoid).
  - Tổng số lớp: 10+.

- **Huấn luyện:**  
  - Loss: `binary_crossentropy`.
  - Optimizer: `Adam`.
  - Batch size: 32.
  - Sử dụng các callback: ModelCheckpoint (lưu mô hình tốt nhất), EarlyStopping (dừng sớm khi không cải thiện), ReduceLROnPlateau (giảm learning rate khi cần).

- **Fine-tune:**  
  - Sau khi huấn luyện cơ bản, tiếp tục fine-tune với learning rate thấp hơn, patience cao hơn, lưu mô hình tốt nhất.

- **Dự đoán:**  
  - Đầu ra là xác suất, nếu >0.5 là Dog, ngược lại là Cat.
  - Hiển thị confidence bar và thông báo mức độ tin cậy.

## Cách sử dụng

1. **Cài đặt thư viện:**
   ```sh
   pip install -r requirements.txt
   ```

2. **Chạy ứng dụng web:**
   ```sh
   streamlit run app.py
   ```
   Truy cập địa chỉ hiển thị trên terminal.

3. **Huấn luyện lại mô hình (nếu muốn):**
   - Mở và chạy các cell trong `Build_model.ipynb`.
   - File mô hình tốt nhất sẽ được lưu tại `fine_tuned_best.h5`.

4. **Deploy lên Render:**
   - Đảm bảo đã có file `render.yaml`.
   - Đẩy code lên Render và cấu hình theo hướng dẫn.

## Tác giả

- duchiep2512

## License

MIT (hoặc bổ sung nếu bạn muốn)