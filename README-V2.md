# rv_net

File new.ipynb là bản chỉnh sửa từ file 2_3_1_HARPS_Linear_FC_CN_June10_2023.ipynb để phù hợp với Tensorflow v2.

Phiên bản này đã test chạy trên Debian Linux 12, với Python 3.11.2

## Steps

1. Tạo môi trường Python ảo với Python 3.11 (thay vì dùng conda phức tạp):
```
 $ mkdir Astro
 $ cd Astro
 $ python3.11 -m venv myvenv
 $ source myvenv/bin/activate
```
2. Download file rvnet-v2.tar.gz vào ~/Downloads và giải nén:
```
 $ tar xvzf ~/Downloads/rvnet-v2.tar.gz
```
Khi này trong thư mục Astro sẽ có thư mục con rvnet-v2

3. Install các dependencies:
```
 $ cd rvnet-v2
 $ pip install -r requirements.txt
```

4. Chạy Jupyter từ trong môi trường ảo:
```
 $ jupyter lab
```

5. Chạy tuần tự các cells trong Jupyter notebook new.ipynb

## CCF raw data to NPZ
1. We need to install mpyfit: https://github.com/evertrol/mpyfit
```
$ git clone https://github.com/evertrol/mpyfit
$ cd mpyfit
$ python setup.py install --user
$ python setup.py build_ext --inplace
$ cp -r mpyfit ../
```
Sau bước này, cần kiểm tra xem dưới thư mục rvnet-v2 có thư mục mpyfit chưa. thư mục mpyfit này phải có file __init__.py

2. Run the script
```
$ python FITS_Preprocess.py
```
This script creates numpy files (.npz) ready for the next steps (create TF files)
