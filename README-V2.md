# rv_net

File new.ipynb là bản chỉnh sửa từ file 2_3_1_HARPS_Linear_FC_CN_June10_2023.ipynb để phù hợp với Tensorflow v2.

Phiên bản này đã test chạy trên Debian Linux 12, với Python 3.11.2

## Steps

1. Tạo môi trường Python ảo với Python 3.11 (thay vì dùng conda phức tạp):

```
 mkdir Astro
 cd Astro
 python3.11 -m venv myvenv
 source myvenv/bin/activate
```

1. Clone repo:

```
 git clone https://github.com/sondo/rvnet-v2.git
```

Khi này trong thư mục Astro sẽ có thư mục con rvnet-v2

1. Install các dependencies:

```
 cd rvnet-v2
 pip install -r requirements.txt
```

1. Chạy Jupyter từ trong môi trường ảo:

```
 jupyter lab
```

1. Chạy tuần tự các cells trong Jupyter notebook new.ipynb

## CCF raw data to NPZ

1. We need to install mpyfit: <https://github.com/evertrol/mpyfit>

For MAC: run ``` pip install -e . ``` (don't forget . et the end)

```
git clone https://github.com/evertrol/mpyfit
cd mpyfit
python setup.py install --user
python setup.py build_ext --inplace
cp -r mpyfit ../
```

Sau bước này, cần kiểm tra xem dưới thư mục rvnet-v2 có thư mục mpyfit chưa. thư mục mpyfit này phải có file __init__.py

1. Run the script
Clean up bad observations (cloud-free <99% or rv_diff_extinction < 0.1m/s)

```
python clean_up.py
```

All bad observations will be changed to .stif and so be ignored in the next steps

- Note: to reverse the change, we need to run:

```
./stif2fits.sh
```

After that we can convert CCF raw data to NPZ files

```
python PrepareData.py
```

This script creates numpy files (.npz) ready for the next steps (create TF files)
