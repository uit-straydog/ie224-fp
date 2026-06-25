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
# Đề tài tập trung phân tích các yếu tố có mối liên hệ với giá nhà và xây dựng
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
# airconditioning, prefarea, furnishingstatus. Các biến này sẽ được encode ở bước tiền xử lý.

# %%
df.isnull().sum()

# %% [markdown]
# Không có missing values, đây là một bộ dữ liệu sạch.

# %%
df.describe().round(2)

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
    print(col, ':', df[col].unique().tolist())

# %% [markdown]
# Các giá trị cụ thể từng cột chữ:
# - mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea: yes/no
# - furnishingstatus: furnished, semi-furnished, unfurnished

# %% [markdown]
# ## 3. Phương pháp phân tích
#
# **Bước 1 - Kiểm tra chất lượng dữ liệu:** nhóm đọc file `Housing.csv`, kiểm tra kích thước dữ liệu, kiểu dữ liệu, missing values, dòng trùng lặp và thống kê mô tả. Mục tiêu của bước này là xác định dữ liệu có đủ sạch để phân tích hay cần xử lý thêm.
#
# **Bước 2 - Phân tích thăm dò dữ liệu (EDA):** nhóm quan sát phân phối của `price` và `area`, dùng boxplot để phát hiện outlier theo quy tắc IQR, phân tích tương quan giữa các biến số và so sánh giá nhà theo các nhóm biến phân loại. Với biến số, Pearson correlation được dùng như góc nhìn tuyến tính ban đầu; Spearman correlation được bổ sung vì `price`, `area`, `bathrooms` và `stories` có lệch phải hoặc outlier. Các hệ số tương quan chỉ được diễn giải như mối liên hệ mô tả, không phải quan hệ nhân quả.
#
# **Bước 3 - Chính sách outlier:** nhóm không loại outlier ngay vì các giá trị giá nhà, diện tích hoặc số tầng cao vẫn có thể hợp lệ trong dữ liệu bất động sản. Tuy nhiên, để tránh kết luận phụ thuộc quá mạnh vào các điểm cực trị, nhóm thêm kiểm tra độ nhạy bằng mô hình hồi quy với `log(price)` và so sánh với mô hình trên giá gốc.
#
# **Bước 4 - Tiền xử lý và thiết kế thực nghiệm:** dữ liệu được chia train/test theo tỷ lệ 80/20 với `random_state = 42`. Các biến phân loại được mã hóa bằng `OneHotEncoder` trong `Pipeline/ColumnTransformer`, nên encoder chỉ được fit trên tập train hoặc trên từng fold trong cross-validation. Cách này nhất quán hơn so với encode toàn bộ dữ liệu trước khi chia train/test.
#
# **Bước 5 - Mô hình và đánh giá:** Linear Regression được chọn như một mô hình nền có tính diễn giải cao, không phải vì tương quan đơn biến tự chứng minh quan hệ tuyến tính đầy đủ. Nhóm so sánh mô hình này với baseline dự đoán giá trung bình và mô hình log-linear `log(price)`. Chất lượng dự đoán được đánh giá bằng `MAE`, `RMSE`, `R²`, tỷ lệ sai số so với giá trung bình tập test và 5-fold cross-validation.
#
# **Bước 6 - Kiểm tra giả định hồi quy:** sau khi huấn luyện, nhóm kiểm tra hệ số chuẩn hóa, biểu đồ giá thực tế so với giá dự đoán, histogram phần dư, biểu đồ residuals vs fitted và Q-Q plot. Các kiểm tra này giúp đánh giá nhanh tính tuyến tính, phân phối phần dư và dấu hiệu phương sai thay đổi.
#

# %% [markdown]
# ![Quy trình phân tích dữ liệu](./figures/data_analysis_workflow.png)
#

# %%
import os
import sys
import subprocess
import warnings
from pathlib import Path

workspace_candidates = [
    Path.cwd() / "requirements.txt",
    Path.cwd() / "ie224-fp" / "requirements.txt",
]
requirements_path = next((path for path in workspace_candidates if path.exists()), None)
if requirements_path is None:
    raise FileNotFoundError("Không tìm thấy requirements.txt để cài đặt thư viện.")

mpl_dir = Path.cwd() / ".mplconfig"
mpl_dir.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

required_modules = {
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "sklearn": "scikit-learn",
    "scipy": "scipy",
}
missing_packages = []

for module_name, package_name in required_modules.items():
    try:
        __import__(module_name)
    except ModuleNotFoundError:
        missing_packages.append(package_name)

if missing_packages:
    print("Đang cài các thư viện còn thiếu:", ", ".join(missing_packages))
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-r",
        str(requirements_path),
    ])

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# %%
duplicate_count = df.duplicated().sum()
print(f"Số dòng bị trùng lặp: {duplicate_count}")

# %% [markdown]
# ## 4. Phân tích thăm dò dữ liệu
#
# Phần này tập trung quan sát phân phối của các biến quan trọng, mức độ tương quan giữa các biến số và sự khác biệt về giá nhà theo từng nhóm thuộc tính.
#

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(df["price"], kde=True, ax=axes[0], color="#1f77b4")
axes[0].set_title("Phân phối giá nhà")
axes[0].set_xlabel("price")

sns.boxplot(x=df["price"], ax=axes[1], color="#9ecae1")
axes[1].set_title("Boxplot của giá nhà")
axes[1].set_xlabel("price")

plt.tight_layout()
plt.show()

# %% [markdown]
# Nhìn vào biểu đồ trên, có thể thấy biến `price` có độ phân tán khá lớn và xuất hiện một số giá trị cao vượt trội so với phần còn lại. Điều này cho thấy giá nhà trong bộ dữ liệu không phân bố hoàn toàn đồng đều và có dấu hiệu tồn tại outlier.
#

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(df["area"], kde=True, ax=axes[0], color="#2ca02c")
axes[0].set_title("Phân phối diện tích nhà")
axes[0].set_xlabel("area")

sns.boxplot(x=df["area"], ax=axes[1], color="#a1d99b")
axes[1].set_title("Boxplot của diện tích")
axes[1].set_xlabel("area")

plt.tight_layout()
plt.show()

# %% [markdown]
# Biến `area` có xu hướng lệch phải, nghĩa là phần lớn căn nhà có diện tích vừa phải, trong khi chỉ có một số ít căn nhà có diện tích rất lớn. Đặc điểm này thường làm cho giá nhà tăng mạnh ở nhóm có diện tích cao.
#

# %%
numeric_cols = ["price", "area", "bedrooms", "bathrooms", "stories", "parking"]
pearson_corr_matrix = df[numeric_cols].corr(method="pearson")
spearman_corr_matrix = df[numeric_cols].corr(method="spearman")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.heatmap(pearson_corr_matrix, annot=True, cmap="Blues", fmt=".2f", ax=axes[0])
axes[0].set_title("Pearson correlation")

sns.heatmap(spearman_corr_matrix, annot=True, cmap="Greens", fmt=".2f", ax=axes[1])
axes[1].set_title("Spearman correlation")

plt.tight_layout()
plt.show()

corr_compare_df = pd.DataFrame(
    {
        "Pearson với price": pearson_corr_matrix["price"],
        "Spearman với price": spearman_corr_matrix["price"],
    }
).sort_values(by="Spearman với price", ascending=False)

corr_matrix = pearson_corr_matrix
corr_compare_df

# %% [markdown]
# Pearson correlation giúp nhóm xem nhanh quan hệ tuyến tính giữa các biến số và `price`. Vì `price`, `area`, `bathrooms` và `stories` có phân phối lệch hoặc outlier, Spearman correlation được bổ sung để xem quan hệ đơn điệu dựa trên thứ hạng, ít nhạy hơn với giá trị cực trị. Cả hai bảng tương quan chỉ phản ánh mối liên hệ mô tả trong dữ liệu, không chứng minh biến nào gây ra thay đổi giá nhà.

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.scatterplot(data=df, x="area", y="price", ax=axes[0], color="#d62728")
axes[0].set_title("Mối quan hệ giữa diện tích và giá nhà")

sns.scatterplot(data=df, x="bathrooms", y="price", ax=axes[1], color="#9467bd")
axes[1].set_title("Mối quan hệ giữa số phòng tắm và giá nhà")

plt.tight_layout()
plt.show()

# %% [markdown]
# Từ scatter plot có thể thấy giá nhà có xu hướng tăng khi diện tích tăng. Bên cạnh đó, số phòng tắm nhiều hơn cũng thường đi kèm với mức giá cao hơn, dù dữ liệu vẫn có độ phân tán tương đối lớn.
#

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(data=df, x="airconditioning", y="price", hue="airconditioning", ax=axes[0], palette="Set2", legend=False)
axes[0].set_title("Giá nhà theo tình trạng điều hòa")
axes[0].set_xlabel("airconditioning")
axes[0].set_ylabel("price")

sns.boxplot(data=df, x="furnishingstatus", y="price", hue="furnishingstatus", ax=axes[1], palette="Set3", legend=False)
axes[1].set_title("Giá nhà theo tình trạng nội thất")
axes[1].set_xlabel("furnishingstatus")
axes[1].set_ylabel("price")

plt.tight_layout()
plt.show()

# %% [markdown]
# Các căn nhà có `airconditioning = yes` thường có mức giá cao hơn nhóm không có điều hòa. Ngoài ra, nhóm `furnished` cũng có xu hướng có giá cao hơn so với `semi-furnished` và `unfurnished`, cho thấy tiện nghi và mức độ hoàn thiện nội thất có mối liên hệ với giá bán.

# %%
categorical_cols = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus",
]

category_price_summary = pd.concat(
    [
        df.groupby(col)["price"]
        .agg(count="size", mean_price="mean", median_price="median")
        .reset_index()
        .rename(columns={col: "group"})
        .assign(variable=col)
        for col in categorical_cols
    ],
    ignore_index=True,
)

category_price_summary = category_price_summary[
    ["variable", "group", "count", "mean_price", "median_price"]
].sort_values(["variable", "group"])

category_price_summary_display = category_price_summary.copy()
category_price_summary_display["mean_price"] = category_price_summary_display["mean_price"].map(lambda value: f"{value:,.0f}")
category_price_summary_display["median_price"] = category_price_summary_display["median_price"].map(lambda value: f"{value:,.0f}")
category_price_summary_display

# %%
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="prefarea", y="price", hue="prefarea", palette="Set2", legend=False)
plt.title("Giá nhà theo vị trí ưu tiên (prefarea)")
plt.xlabel("prefarea")
plt.ylabel("price")
plt.show()

# %% [markdown]
# Bảng thống kê theo nhóm và boxplot bổ sung cho thấy nhóm `prefarea = yes` có giá trung bình và trung vị cao hơn nhóm `prefarea = no`. Vì vậy, `prefarea` là một biến phân loại đáng chú ý khi phân tích mối liên hệ với giá nhà.

# %%
outlier_summary = []
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    outlier_summary.append({
        "column": col,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": outlier_count,
    })

outlier_df = pd.DataFrame(outlier_summary)
outlier_df

# %% [markdown]
# Bảng trên cho thấy một số biến như `price`, `area` hoặc `stories` có xuất hiện outlier theo quy tắc IQR. Nhóm giữ lại các quan sát này vì trong dữ liệu bất động sản, nhà giá cao, diện tích lớn hoặc nhiều tầng vẫn có thể là trường hợp hợp lệ. Tuy nhiên, việc giữ outlier có thể ảnh hưởng đến hồi quy tuyến tính, nên phần mô hình bên dưới bổ sung kiểm tra độ nhạy bằng cách so sánh mô hình giá gốc với mô hình dùng `log(price)`.

# %% [markdown]
# Từ kết quả phân tích sơ bộ, có thể thấy một số thuộc tính như `area`, `bathrooms`, `airconditioning` và `prefarea` có mối liên hệ với giá nhà. Nhóm sử dụng Linear Regression như một mô hình nền dễ diễn giải để lượng hóa các mối liên hệ này, sau đó kiểm tra chất lượng dự đoán, phần dư và so sánh thêm với mô hình `log(price)` để đánh giá độ nhạy với phân phối lệch phải/outlier.

# %% [markdown]
# ## 5. Xây dựng mô hình hồi quy
#
# Ở bước này, nhóm chia dữ liệu thành tập huấn luyện và tập kiểm tra trước, sau đó đưa các bước mã hóa biến phân loại vào `Pipeline/ColumnTransformer`. Thiết kế này giúp quá trình tiền xử lý được fit trên dữ liệu huấn luyện hoặc từng fold cross-validation, phù hợp hơn với quy trình đánh giá mô hình.

# %%
target_col = "price"
numeric_features = ["area", "bedrooms", "bathrooms", "stories", "parking"]
categorical_features = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus",
]

X = df.drop(columns=[target_col])
y = df[target_col]

feature_plan_df = pd.DataFrame(
    [
        {"Nhóm biến": "Numeric", "Biến": ", ".join(numeric_features), "Xử lý": "Giữ giá trị gốc trong mô hình chính"},
        {"Nhóm biến": "Categorical", "Biến": ", ".join(categorical_features), "Xử lý": "OneHotEncoder(drop='first') trong Pipeline"},
        {"Nhóm biến": "Target", "Biến": target_col, "Xử lý": "Giá gốc; kiểm tra độ nhạy thêm với log(price)"},
    ]
)
feature_plan_df


# %%
def make_regression_pipeline(scale_all_features=False):
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric_features),
            (
                "categorical",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ],
        verbose_feature_names_out=False,
    )

    steps = [("preprocess", preprocessor)]
    if scale_all_features:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", LinearRegression()))
    return Pipeline(steps)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

linear_model = make_regression_pipeline(scale_all_features=False)
linear_model.fit(X_train, y_train)
with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
    y_pred = linear_model.predict(X_test)

log_price_model = TransformedTargetRegressor(
    regressor=make_regression_pipeline(scale_all_features=False),
    func=np.log1p,
    inverse_func=np.expm1,
)
log_price_model.fit(X_train, y_train)
with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
    y_pred_log = log_price_model.predict(X_test)

baseline_model = DummyRegressor(strategy="mean")
baseline_model.fit(X_train, y_train)
y_pred_baseline = baseline_model.predict(X_test)


# %%
def regression_metrics(model_name, y_true, y_prediction):
    return {
        "Mô hình": model_name,
        "MAE": mean_absolute_error(y_true, y_prediction),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_prediction)),
        "R²": r2_score(y_true, y_prediction),
    }


mean_test_price = y_test.mean()
metric_df = pd.DataFrame(
    [
        regression_metrics("Baseline: dự đoán giá trung bình", y_test, y_pred_baseline),
        regression_metrics("Linear Regression - price gốc", y_test, y_pred),
        regression_metrics("Linear Regression - log(price)", y_test, y_pred_log),
    ]
)
metric_df["MAE / giá TB test"] = metric_df["MAE"] / mean_test_price
metric_df["RMSE / giá TB test"] = metric_df["RMSE"] / mean_test_price

metric_df_display = metric_df.copy()
for col in ["MAE", "RMSE"]:
    metric_df_display[col] = metric_df_display[col].map(lambda value: f"{value:,.0f}")
metric_df_display["R²"] = metric_df_display["R²"].map(lambda value: f"{value:.3f}")
for col in ["MAE / giá TB test", "RMSE / giá TB test"]:
    metric_df_display[col] = metric_df_display[col].map(lambda value: f"{value:.1%}")

metric_df_display

# %%
cv = KFold(n_splits=5, shuffle=True, random_state=42)
models_for_cv = {
    "Baseline: dự đoán giá trung bình": DummyRegressor(strategy="mean"),
    "Linear Regression - price gốc": make_regression_pipeline(scale_all_features=False),
    "Linear Regression - log(price)": TransformedTargetRegressor(
        regressor=make_regression_pipeline(scale_all_features=False),
        func=np.log1p,
        inverse_func=np.expm1,
    ),
}

cv_rows = []
for model_name, estimator in models_for_cv.items():
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        cv_mae = -cross_val_score(estimator, X, y, cv=cv, scoring="neg_mean_absolute_error")
        cv_rmse = np.sqrt(-cross_val_score(estimator, X, y, cv=cv, scoring="neg_mean_squared_error"))
        cv_r2 = cross_val_score(estimator, X, y, cv=cv, scoring="r2")
    cv_rows.append({
        "Mô hình": model_name,
        "CV MAE mean": cv_mae.mean(),
        "CV MAE std": cv_mae.std(),
        "CV RMSE mean": cv_rmse.mean(),
        "CV RMSE std": cv_rmse.std(),
        "CV R² mean": cv_r2.mean(),
        "CV R² std": cv_r2.std(),
    })

cv_df = pd.DataFrame(cv_rows)
cv_df_display = cv_df.copy()
for col in ["CV MAE mean", "CV MAE std", "CV RMSE mean", "CV RMSE std"]:
    cv_df_display[col] = cv_df_display[col].map(lambda value: f"{value:,.0f}")
for col in ["CV R² mean", "CV R² std"]:
    cv_df_display[col] = cv_df_display[col].map(lambda value: f"{value:.3f}")

cv_df_display

# %% [markdown]
# Bảng đánh giá so sánh ba mức: baseline dự đoán giá trung bình, Linear Regression trên giá gốc và Linear Regression với biến mục tiêu `log(price)`. Mô hình `log(price)` đóng vai trò kiểm tra độ nhạy với phân phối lệch phải và outlier của giá nhà. Ngoài kết quả trên tập test, nhóm dùng 5-fold cross-validation để kiểm tra độ ổn định của mô hình trên nhiều cách chia dữ liệu.

# %%
feature_names = linear_model.named_steps["preprocess"].get_feature_names_out()
original_coefs = linear_model.named_steps["model"].coef_

standardized_model = make_regression_pipeline(scale_all_features=True)
standardized_model.fit(X_train, y_train)
standardized_coefs = standardized_model.named_steps["model"].coef_

coef_df = pd.DataFrame(
    {
        "Biến": feature_names,
        "Hệ số gốc": original_coefs,
        "Hệ số chuẩn hóa": standardized_coefs,
    }
).sort_values(by="Hệ số chuẩn hóa", key=lambda s: s.abs(), ascending=False)

coef_df_display = coef_df.copy()
coef_df_display["Hệ số gốc"] = coef_df_display["Hệ số gốc"].map(lambda value: f"{value:,.0f}")
coef_df_display["Hệ số chuẩn hóa"] = coef_df_display["Hệ số chuẩn hóa"].map(lambda value: f"{value:,.0f}")
coef_df_display

# %% [markdown]
# Bảng hệ số hồi quy được lấy từ pipeline đã fit trên tập train. Hệ số gốc cho biết mức thay đổi dự đoán theo đơn vị ban đầu của từng biến sau khi encode. Vì các biến có thang đo khác nhau, nhóm dùng thêm `Hệ số chuẩn hóa` để so sánh tương đối giữa các biến. Đây vẫn là diễn giải mô hình tuyến tính trên dữ liệu quan sát, nên chỉ nên hiểu là mối liên hệ với giá dự đoán, không phải quan hệ nhân quả.

# %%
plt.figure(figsize=(7, 7))
sns.scatterplot(x=y_test, y=y_pred, color="#1f77b4")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--")
plt.xlabel("Giá thực tế")
plt.ylabel("Giá dự đoán")
plt.title("So sánh giá thực tế và giá dự đoán")
plt.show()

# %%
residuals = y_test - y_pred
fitted_values = y_pred

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.histplot(residuals, kde=True, color="#ff7f0e", ax=axes[0])
axes[0].set_title("Phân phối phần dư")
axes[0].set_xlabel("Residuals")

sns.scatterplot(x=fitted_values, y=residuals, color="#1f77b4", ax=axes[1])
axes[1].axhline(0, color="red", linestyle="--")
axes[1].set_title("Residuals vs Fitted")
axes[1].set_xlabel("Giá dự đoán")
axes[1].set_ylabel("Phần dư")

stats.probplot(residuals, dist="norm", plot=axes[2])
axes[2].set_title("Q-Q plot phần dư")

plt.tight_layout()
plt.show()

# %% [markdown]
# Biểu đồ phần dư giúp kiểm tra nhanh giả định của hồi quy tuyến tính trên mô hình giá gốc. Histogram và Q-Q plot cho biết phần dư có gần phân phối chuẩn hay không, còn biểu đồ `Residuals vs Fitted` giúp quan sát phần dư có phân tán ngẫu nhiên quanh 0 hay xuất hiện dấu hiệu phương sai thay đổi. Nếu phần dư có dạng phễu hoặc lệch mạnh, kết quả `log(price)` ở bảng trên là một kiểm tra độ nhạy quan trọng.

# %% [markdown]
# ## 6. Kết quả phân tích
#
# Qua quá trình phân tích, nhóm nhận thấy diện tích, số phòng tắm, vị trí ưu tiên và các tiện nghi như điều hòa có mối liên hệ tích cực với giá nhà. Pearson và Spearman correlation được dùng như phân tích mô tả ban đầu, còn mô hình hồi quy tuyến tính được dùng như baseline có tính diễn giải để dự đoán giá nhà.
#
# Kết quả đánh giá cho thấy mô hình hồi quy tuyến tính tốt hơn baseline dự đoán giá trung bình. Kiểm tra thêm với `log(price)` giúp đánh giá mô hình có nhạy với phân phối lệch phải và outlier của giá nhà hay không. Các biểu đồ phần dư cho thấy cần tiếp tục thận trọng với giả định tuyến tính, phân phối chuẩn của phần dư và phương sai không đổi khi sử dụng mô hình hồi quy tuyến tính cho bộ dữ liệu này.

# %% [markdown]
# ## 7. Kết luận
#
# Bài toán phân tích và dự đoán giá nhà cho thấy giá nhà có mối liên hệ với nhiều thuộc tính khác nhau, trong đó diện tích, số phòng tắm, vị trí ưu tiên và một số tiện nghi là các biến nổi bật trong bộ dữ liệu. Thông qua phân tích thăm dò, so sánh Pearson/Spearman, pipeline tiền xử lý, đánh giá baseline, cross-validation, kiểm tra `log(price)` và phân tích phần dư, nhóm có thể trình bày quy trình phân tích chặt chẽ hơn.
#
# Trong tương lai, nhóm có thể thử thêm Ridge, Lasso, Random Forest hoặc robust regression để so sánh hiệu quả dự đoán và kiểm tra thêm độ ổn định của kết luận khi dữ liệu có outlier.
