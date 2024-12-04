import pandas as pd
import statsmodels.api as sm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 데이터 불러오기
def load_data(file_path):
    # CSV 파일을 로드하고 데이터프레임 반환
    return pd.read_csv(file_path)

# 2. 다중회귀분석 함수 정의
def run_ols_regression(X, y):
    # 독립변수(X), 종속변수(y)를 사용하여 OLS 회귀 분석 실행
    X = sm.add_constant(X)  # 상수항 추가
    model = sm.OLS(y, X).fit()
    print(model.summary())  # 결과 출력
    return model

# 3. 잔차 분석 시각화
def plot_residual_analysis(model,title):
    # 잔차 분석을 위한 시각화
    residuals = model.resid
    fitted_values = model.fittedvalues

    # 잔차 vs 예측값
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=fitted_values, y=residuals)
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title('Residuals vs Fitted Values '+ title)
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.show()

    # 잔차 분포 (히스토그램 & QQ Plot)
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, bins=30)
    plt.title('Distribution of residuals'+ title)
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.show()

# 4. 이상치 탐지 및 제거
def remove_outliers(data, residuals, threshold=3):
    # 잔차 기준으로 이상치를 탐지하고 제거
    residual_std = residuals.std()
    outlier_mask = np.abs(residuals) > (threshold * residual_std)
    outlier_indices = residuals[outlier_mask].index

    # 이상치 제거
    data_cleaned = data[~outlier_mask].copy()
    print(f"Number of outliers removed: {len(outlier_indices)}")
    print(f"Data size after outlier removal: {data_cleaned.shape}")
    return data_cleaned


# 5. 실제값과 예측값을 비교하는 플롯 생성
def plot_actual_vs_predicted(X, y, model, title):
    """
    Parameters:
    - X: 독립변수 (Pandas DataFrame or Series)
    - y: 종속변수 (Pandas Series)
    - model: Statsmodels 회귀 모델 객체
    - title: 플롯 제목 (str)
    """
    # 상수항 포함된 독립변수 생성
    X_with_const = sm.add_constant(X)
    
    # 예측값 계산
    predicted = model.predict(X_with_const)

    # 플롯 생성
    plt.figure(figsize=(8, 6))
    plt.scatter(y, predicted, color='blue', label='Predicted vs Actual')  # 예측값 vs 실제값
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='Ideal Fit')  # 완벽한 예측선
    plt.title(title)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.show()

# 6. 전체 실행 로직
def main(file_path):
    # 데이터 로드
    data = load_data(file_path)
    
    # Trust -> R&D/GDP (다중회귀분석)
    print("\n--- Trust -> R&D/GDP ---")
    X = data[['Scientist', 'Government', 'Gross_tertiary_education_enrollment']]
    y = data['RnD_ratio']
    model_rnd_ratio = run_ols_regression(X, y)
    plot_residual_analysis(model_rnd_ratio, '(Trust -> R&D/GDP)')

    # 이상치 탐지 및 제거
    data_cleaned = remove_outliers(data, model_rnd_ratio.resid)

    # Trust -> RnD Investment/GDP (클린 데이터 사용)
    print("\n--- Trust -> R&D/GDP (Cleaned Data) ---")
    X_cleaned = data_cleaned[['Scientist', 'Government', 'Gross_tertiary_education_enrollment']]
    y_cleaned = data_cleaned['RnD_ratio']
    model_rnd_ratio_cleaned = run_ols_regression(X_cleaned, y_cleaned)
    plot_actual_vs_predicted(X_cleaned, y_cleaned, model_rnd_ratio_cleaned, title="Actual vs Predicted (Trust -> R&D/GDP)")

    # RnD_ratio -> Innovation (overall index) (다중회귀분석)
    print("\n--- R&D/GDP -> Innovation (overall index) ---")
    X_innovation = data_cleaned[['RnD_ratio', 'Gross_tertiary_education_enrollment']]
    y_innovation = data_cleaned['Innovation_overall_index']
    model_innovation = run_ols_regression(X_innovation, y_innovation)
    plot_residual_analysis(model_innovation, '(R&D/GDP -> Innovation)')
    plot_actual_vs_predicted(X_innovation, y_innovation, model_innovation, title="Actual vs Predicted (R&D/GDP -> Innovation)")

    # RnD_ratio -> Research & Developement (다중회귀분석)
    print("\n--- R&D/GDP -> Research & Developement ---")
    X_RnD = data_cleaned[['RnD_ratio', 'Gross_tertiary_education_enrollment']]
    y_RnD = data_cleaned['Research_Development']
    model_rnd = run_ols_regression(X_RnD, y_RnD)
    plot_residual_analysis(model_rnd, '(R&D/GDP -> Research & Developement)')
    plot_actual_vs_predicted(X_RnD, y_RnD, model_innovation, title="Actual vs Predicted (R&D/GDP -> Research & Developement)")

    # Trust -> Innovation (overall index) (다중회귀분석)
    print("\n--- Trust -> Innovation (overall index) ---")
    X_innovation_dir = data_cleaned[['Scientist', 'Government', 'Gross_tertiary_education_enrollment']]
    y_innovation_dir = data_cleaned['Innovation_overall_index']
    model_innovation_dir = run_ols_regression(X_innovation_dir, y_innovation_dir)
    plot_residual_analysis(model_innovation_dir, '(Trust -> Innovation)')
    plot_actual_vs_predicted(X_innovation_dir, y_innovation_dir, model_innovation_dir, title="Actual vs Predicted (Trust -> Innovation)")


    # Trust -> Research & Developement (다중회귀분석)
    print("\n--- Trust -> Research & Developement ---")
    X_RnD_dir = data_cleaned[['Scientist', 'Government', 'Gross_tertiary_education_enrollment']]
    y_RnD_dir = data_cleaned['Research_Development']
    model_RnD_dir = run_ols_regression(X_RnD_dir, y_RnD_dir)
    plot_residual_analysis(model_RnD_dir, '(Trust -> Research & Developement)')
    plot_actual_vs_predicted(X_RnD_dir, y_RnD_dir, model_RnD_dir, title="Actual vs Predicted (Trust -> Research & Developement)")
    

# 파일 경로 설정 및 실행
file_path = 'Merged_Data.csv'
main(file_path)
