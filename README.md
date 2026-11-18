# Trashing Detection

Chương trình C++ nhận một video và cho biết có hành vi đổ rác hay không. Nếu có, in ra các khoảng thời gian chứa hành vi đó. Mô hình MoViNet-A0 backbone kèm đầu phân loại đã được huấn luyện được chạy bằng TensorRT, còn video được đọc bằng OpenCV.

## Cấu trúc thư mục

```
.
├── CMakeLists.txt
├── trashing_detection.cpp
├── src/                 # đặt file movinet_a0_dumping.engine vào đây
└── DATASET/
    └── videos/          # đặt video cần kiểm tra vào đây
```

## Dữ liệu
 
Video dùng để huấn luyện và kiểm tra lấy từ bộ dữ liệu MIVIA-IWDD: https://github.com/MiviaLab/IllegalWasteDumping

## Yêu cầu

- Windows, Visual Studio 2022 (workload *Desktop development with C++*), CMake >= 3.23
- GPU NVIDIA, CUDA Toolkit 13.x
- TensorRT 11.4 (bản zip cho Windows)
- OpenCV 5.0 (bản dựng sẵn cho Windows)

Thêm thư mục `bin` của TensorRT, OpenCV và CUDA vào biến môi trường `Path`.

## Cài đặt

**1. Dựng engine từ file ONNX của mô hình.**

```bat
trtexec --onnx=movinet_a0_dumping.onnx --saveEngine=src/movinet_a0_dumping.engine
```

**2. Sửa cấu hình trong `trashing_detection.cpp`** cho phù hợp với máy và mong muốn:

```cpp
const std::string modelPath = "src/movinet_a0_dumping.engine";  // đường dẫn file engine
const std::string videoPath = "DATASET/videos/vid0012.mp4";     // đường dẫn video cần kiểm tra
const float THRESHOLD = 0.5;                                     // ngưỡng tin cậy (0 - 1)
```

**3. Sửa đường dẫn TensorRT trong `CMakeLists.txt`:**

```cmake
set(TRT_DIR "C:/libs/TensorRT-11.4.0.106")
```

**4. Build:**

```bat
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DOpenCV_DIR=C:/libs/opencv/build
cmake --build build --config Release
```

Sửa lại đường dẫn `OpenCV_DIR`, `OpenCV_DIR`  là thư mục chứa file `OpenCVConfig.cmake`.

## Chạy

Chạy từ thư mục gốc của repo vì các đường dẫn trong code là đường dẫn tương đối:

```bat
build\Release\trashing_detection.exe
```

Ví dụ kết quả:

```
Video: DATASET/videos/vid0012.mp4 - 15.8 giay
Cua so: 0s den 10s co xac suat: 0.21
Cua so: 5s den 15s co xac suat: 0.97
Cua so: 5.8s den 15.8s co xac suat: 0.95
Ket qua: Phat hien do rac tai cac khoang thoi gian:
- 5s den 15.8s
```

Mỗi cửa sổ dài 10 giây, hai cửa sổ liền nhau cách nhau 5 giây (chống lấn 5 giây). Cửa sổ nào có xác suất lớn hơn hoặc bằng ngưỡng thì được coi là có đổ rác. Các cửa sổ có đổ rác nằm chồng lên nhau được gộp lại thành một khoảng thời gian.
