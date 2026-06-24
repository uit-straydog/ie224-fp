# %% [markdown]
#
# | | |
# |---|---|
# | **Nhóm** | 4 |
# | **Thành viên** | 23730041 - Dương Thanh Quí |
# | | 24730010 - Phan Tuấn Anh |
# | | 24730004 - Hoàng Quốc Anh |
# | **Lớp** | IE224.F21.CN1.CNTT |
# | **Học phần** | Phân tích dữ liệu |
# | **GVHD** | ThS. Phạm Thế Sơn |
#
# # Đề tài: PHÂN TÍCH VÀ DỰ ĐOÁN GIÁ NHÀ SỬ DỤNG MÔ HÌNH HỒI QUY
#
# ## 1. Giới thiệu
#
# Đề tài tập trung phân tích các yếu tố ảnh hưởng đến giá nhà và xây dựng
# mô hình dự đoán giá dựa trên các đặc trưng như diện tích, số phòng ngủ,
# số phòng tắm, tiện nghi và vị trí.
#
# Bộ dữ liệu được sử dụng được thu thập từ
# [Kaggle](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset).
#
# Mục tiêu: Xây dựng mô hình hồi quy dự đoán giá nhà (`price`) dựa trên
# các thuộc tính còn lại.
#

# %%
import pandas as pd

pd.options.future.infer_string = True
df = pd.read_csv("./data/Housing.csv")

# %% [markdown]
# ## 2. Mô tả bộ dữ liệu

# %%
df.shape

# %% [markdown]
# Bộ dữ liệu gồm 545 dòng, 13 cột.

# %%
df.dtypes

# %% [markdown]
# Bộ dữ liệu gồm 13 cột:
# - 6 cột số (int64): price, area, bedrooms, bathrooms, stories, parking
# - 7 cột chữ (str): mainroad, guestroom, basement, hotwaterheating,
# airconditioning, prefarea, furnishingstatus sẽ được encode ở bước tiền xử lý.

# %%
df.isnull().sum()

# %% [markdown]
# Không có missing values, đây là một bộ dữ liệu sạch.

# %%
df.describe().style.format('{:.2f}')

# %% [markdown]
# - Dataset có 545 dòng (count = 545)
# - Giá nhà (price) trung bình ~4.77 triệu, dao động từ 1.75 triệu đến 13.3 triệu
# - Diện tích (area) trung bình 5150, dao động từ 1650 đến 16200
# - Số phòng ngủ (bedrooms) phổ biến là 3 (median = 3), tối đa 6
# - Số phòng tắm (bathrooms) phổ biến là 1, tối đa 4
# - Số tầng (stories) trung bình ~1.8, tối đa 4
# - Bãi đậu xe (parking) phần lớn là 0 (median = 0), tối đa 3

# %%
df.describe(include='str')

# %% [markdown]
# Nhận xét các cột chữ:
# - 6 cột có 2 giá trị là chữ là: mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea
# - 1 cột có 3 giá trị là chữ là: furnishingstatus

# %%
for col in df.select_dtypes(include='str').columns:
    print(col, ':', df[col].unique())

# %% [markdown]
# Các giá trị cụ thể từng cột chữ:
# - mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea: yes/no
# - furnishingstatus: furnished, semi-furnished, unfurnished
