"""
المشروع النهائي - المرحلة الأولى: تحليل نجاح الطلاب (Student Success Analytics)
الملف: analysis.py
الوصف: السكريبت الشامل والنهائي مع التعليقات العربية قبل كل رسم بياني.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from transformers import pipeline

# ==========================================
# 0. إعداد البيئة العامة للمشروع
# ==========================================
os.makedirs('charts', exist_ok=True)
sns.set_theme(style='whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

print('========================================')
print('1. Loading and Cleaning Data...')
print('========================================')

students = pd.read_csv('Students.csv')
enrollments = pd.read_csv('Enrollments.csv')
quizzes = pd.read_csv('Quizzes.csv')
reviews = pd.read_csv('Reviews.csv')



# ==========================================
# 1. فحص القيم المفقودة
# ==========================================
print('--- Missing Values Check ---')
for name, df in [
    ('Students', students),
    ('Enrollments', enrollments),
    ('Quizzes', quizzes),
    ('Reviews', reviews),
]:
  missing = df.isnull().sum()
  if missing.sum() > 0:
    print(f'-> {name} missing counts:\n{missing[missing > 0]}\n')
  else:
    print(f'-> {name}: No missing values.')

# ==========================================
# معالجة القيم المفقودة (Missing Values Treatment)
# ==========================================
print('\n========================================')
print('Handling Missing Values...')
print('========================================')

# 1. معالجة القيم المفقودة في جدول الطلاب (Students)
if 'Age' in students.columns:
  students['Age'] = students['Age'].fillna(students['Age'].median())

for col in ['Gender', 'Country', 'Occupation']:
  if col in students.columns:
    students[col] = students[col].fillna('Unknown')

# 2. معالجة القيم المفقودة في جدول التسجيلات (Enrollments)
if 'CompletionPercentage' in enrollments.columns:
  enrollments['CompletionPercentage'] = enrollments[
      'CompletionPercentage'
  ].fillna(0)

# 3. معالجة القيم المفقودة في جدول الاختبارات (Quizzes)
if 'Score' in quizzes.columns:
  quizzes['Score'] = quizzes['Score'].fillna(quizzes['Score'].mean())

if 'Attempts' in quizzes.columns:
  quizzes['Attempts'] = quizzes['Attempts'].fillna(1)

# 4. معالجة القيم المفقودة في جدول التقييمات (Reviews)
if 'Rating' in reviews.columns:
  reviews['Rating'] = reviews['Rating'].fillna(reviews['Rating'].mean())

print('Missing values successfully cleaned and handled across all dataframes!')
print('========================================')
# 2. إزالة الصفوف المكررة (Duplicate Rows)
for name, df in [
    ('Students', students),
    ('Enrollments', enrollments),
    ('Quizzes', quizzes),
    ('Reviews', reviews),
]:
  initial_rows = len(df)
  df.drop_duplicates(inplace=True)
  print(
      f'-> Cleaned {name}: Removed {initial_rows - len(df)} duplicate rows.'
  )


# 3. تحويل التواريخ وتصحيح النصوص
students['RegistrationDate'] = pd.to_datetime(students['RegistrationDate'])
enrollments['EnrollmentDate'] = pd.to_datetime(enrollments['EnrollmentDate'])

for col in ['Gender', 'Country', 'Occupation']:
  students[col] = students[col].str.strip()

for col in ['Completed', 'Progress']:
  enrollments[col] = enrollments[col].str.strip()

reviews['Comment'] = reviews['Comment'].str.strip()

# 4. اكتشاف ومعالجة القيم الشاذة (Outliers & Range Check)
print('\n--- Checking Outliers and Ranges ---')
reviews.loc[reviews['Rating'] < 1, 'Rating'] = 1
reviews.loc[reviews['Rating'] > 5, 'Rating'] = 5
enrollments.loc[enrollments['CompletionPercentage'] < 0, 'CompletionPercentage'] = (
    0
)
enrollments.loc[
    enrollments['CompletionPercentage'] > 100, 'CompletionPercentage'
] = 100
quizzes.loc[quizzes['Score'] < 0, 'Score'] = 0
quizzes.loc[quizzes['Score'] > 100, 'Score'] = 100
print('-> Outliers checked and handled successfully.')

# ==========================================
# 2. دمج الجداول (Data Merging)
# ==========================================
print('\n========================================')
print('2. Merging Datasets...')
print('========================================')

df_merged = pd.merge(students, enrollments, on='StudentID', how='left')
df_merged = pd.merge(
    df_merged, quizzes, on=['StudentID', 'CourseID'], how='left'
)
df_merged = pd.merge(
    df_merged, reviews, on=['StudentID', 'CourseID'], how='left'
)

df_merged.to_csv('full_merged_dataset.csv', index=False)
print(f'-> Final merged dataset shape: {df_merged.shape}')

# ==========================================
# 3. هندسة الميزات (Feature Engineering)
# ==========================================
print('\n========================================')
print('3. Feature Engineering...')
print('========================================')

df_merged['RegistrationYear'] = df_merged['RegistrationDate'].dt.year
df_merged['RegistrationMonth'] = df_merged['RegistrationDate'].dt.month

bins = [0, 20, 30, 40, 100]
labels = ['Under 20', '20-30', '31-40', '40+']
df_merged['AgeGroup'] = pd.cut(
    df_merged['Age'], bins=bins, labels=labels, right=False
)

conditions = [
    df_merged['CompletionPercentage'] == 100,
    (df_merged['CompletionPercentage'] >= 50)
    & (df_merged['CompletionPercentage'] < 100),
    df_merged['CompletionPercentage'] < 50,
]
choices = ['Completed', 'In Progress', 'At Risk']
df_merged['StudentStatus'] = np.select(
    conditions, choices, default='Unknown'
)
df_merged['Is_Success'] = (df_merged['StudentStatus'] == 'Completed').astype(int)
df_merged['Is_Failure'] = (df_merged['StudentStatus'] == 'At Risk').astype(int)
df_merged['Is_Completed'] = (df_merged['StudentStatus'] == 'Completed').astype(
    int
)

print('-> Feature engineering completed successfully.')

# ==========================================
# 4. التحليل الإحصائي، GroupBy و Pivot Tables
# ==========================================
print('\n========================================')
print('4. Statistical Analysis, GroupBy & Pivot Tables...')
print('========================================')

country_grouped = (
    df_merged.groupby('Country')
    .agg(
        Total_Students=('StudentID', 'nunique'),
        Avg_Completion=('CompletionPercentage', 'mean'),
        Avg_Score=('Score', 'mean'),
    )
    .reset_index()
)
print('-> GroupBy Results (Country Stats Sample):')
print(country_grouped.head())

pivot_table_result = pd.pivot_table(
    df_merged,
    values='Score',
    index='AgeGroup',
    columns='StudentStatus',
    aggfunc='mean',
    fill_value=0,
)
print('-> Pivot Table Results:')
print(pivot_table_result)

# ==========================================
# Step 1: Calculating Total Students
# ==========================================
print('\n========================================')
print('Step 1: Calculating Total Students...')
print('========================================')

total_students_unique = df_merged['StudentID'].nunique()


print(f'-> Total Unique Students: {total_students_unique}')

# ==========================================
# Step 2: Calculating Total Countries
# ==========================================
print('\n========================================')
print('Step 2: Calculating Total Countries...')
print('========================================')

total_countries = df_merged['Country'].nunique()

print(f'-> Total Unique Countries: {total_countries}')


# ==========================================
# Step 3: Calculating Average Age
# ==========================================
print('\n========================================')
print('Step 3: Calculating Average Age...')
print('========================================')

avg_age = df_merged['Age'].mean()

print(f'-> Average Age: {avg_age:.2f} years')

# ==========================================
# Step 4: Calculating Average Completion Percentage
# ==========================================
print('\n========================================')
print('Step 4: Calculating Average Completion Percentage...')
print('========================================')

avg_completion = df_merged['CompletionPercentage'].mean()

print(f'-> Average Completion Percentage: {avg_completion:.2f}%')



# ==========================================
# Step 5: Calculating Success Rate
# ==========================================
print('\n========================================')
print('Step 5: Calculating Success Rate...')
print('========================================')

success_rate = (df_merged['Is_Success'].mean()) * 100

print(f'-> Success Rate: {success_rate:.2f}%')

# ==========================================
# Step 6: Calculating Failure Rate
# ==========================================
print('\n========================================')
print('Step 6: Calculating Failure Rate...')
print('========================================')

failure_rate = (df_merged['Is_Failure'].mean()) * 100

print(f'-> Failure Rate: {failure_rate:.2f}%')

# ==========================================
# Step 7: Calculating Student Status Distribution (Completed, In Progress, At Risk)
# ==========================================
print('\n========================================')
print('Step 7: Calculating Student Status Distribution...')
print('========================================')

status_counts = df_merged['StudentStatus'].value_counts(normalize=True) * 100

for status, percentage in status_counts.items():
  print(f'-> {status}: {percentage:.2f}%')


# ==========================================
# Step 8: Calculating Average Quiz Attempts
# ==========================================
print('\n========================================')
print('Step 8: Calculating Average Quiz Attempts...')
print('========================================')

avg_attempts = df_merged['Attempts'].mean()

print(f'-> Average Quiz Attempts: {avg_attempts:.2f}')

# ==========================================
# Step 9: Calculating Average Course Rating
# ==========================================
print('\n========================================')
print('Step 9: Calculating Average Course Rating...')
print('========================================')

avg_rating = df_merged['Rating'].mean()

print(f'-> Average Course Rating: {avg_rating:.2f} / 5')


# ==========================================
# Step 10: Finding Top 10 Students
# ==========================================
print('\n========================================')
print('Step 10: Top 10 Students...')
print('========================================')

top_students = (
    df_merged.groupby(['StudentID', 'Name'])[['CompletionPercentage', 'Score']]
    .mean()
    .reset_index()
)
top_students = top_students.sort_values(
    by=['CompletionPercentage', 'Score'], ascending=False
).head(10)

print(top_students)

# ==========================================
# Step 11: Most Successful Countries
# ==========================================
print('\n========================================')
print('Step 11: Most Successful Countries...')
print('========================================')

success_by_country = (
    df_merged.groupby('Country')['Is_Success'].mean().reset_index()
)
success_by_country['Success_Rate_%'] = (
    success_by_country['Is_Success'] * 100
)
success_by_country = success_by_country.sort_values(
    by='Success_Rate_%', ascending=False
)

print(success_by_country)


# ==========================================
# Step 12: Most Successful Age Groups
# ==========================================
print('\n========================================')
print('Step 12: Most Successful Age Groups...')
print('========================================')

bins = [0, 20, 30, 40, 100]
labels = ['Under 20', '20-30', '31-40', '40+']
df_merged['AgeGroup'] = pd.cut(
    df_merged['Age'], bins=bins, labels=labels, right=False
)

success_by_age = (
    df_merged.groupby('AgeGroup', observed=False)['Is_Success']
    .mean()
    .reset_index()
)
success_by_age['Success_Rate_%'] = success_by_age['Is_Success'] * 100
success_by_age = success_by_age.sort_values(
    by='Success_Rate_%', ascending=False
)

print(success_by_age)

# ==========================================
# Step 13: Countries with Highest Average Rating
# ==========================================
print('\n========================================')
print('Step 13: Countries with Highest Average Rating...')
print('========================================')

rating_by_country = (
    df_merged.groupby('Country')['Rating']
    .mean()
    .reset_index()
)
rating_by_country = rating_by_country.sort_values(
    by='Rating', ascending=False
)

print(rating_by_country)





# 1. قراءة ملف التسجيلات بشكل صريح
df_enrollments = pd.read_csv('Enrollments.csv')

# 2. تجميع البيانات حسب الدورة وحساب متوسط نسبة الإكمال
course_summary = df_enrollments.groupby('CourseID')['CompletionPercentage'].mean().reset_index()

# 3. الدورة ذات أعلى معدل إكمال (الأكثر نجاحاً)
top_course = course_summary.loc[course_summary['CompletionPercentage'].idxmax()]

# 4. الدورة ذات أقل معدل إكمال (الأكثر صعوبة)
hardest_course = course_summary.loc[course_summary['CompletionPercentage'].idxmin()]

# 5. طباعة النتائج بوضوح
print("--- Top-Course---")
print(f"Course ID: {top_course['CourseID']}")
print(f"Completion Rate: {top_course['CompletionPercentage']:.2f}%\n")

print("--- Hardest-Course---")
print(f"Course ID: {hardest_course['CourseID']}")
print(f"Completion Rate: {hardest_course['CompletionPercentage']:.2f}%")

print('========================================')

#==========================================
#5. توليد جميع الرسوم البيانية (All Visualizations)
#==========================================
print('5. Generating All Visualizations...')
print('========================================')

# 1. رسم مخطط الدائرة النسبية لحالة الطلاب الثلاثة (Pie Chart)
status_counts = df_merged['StudentStatus'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(
    status_counts,
    labels=status_counts.index,  # سيأخذ الأسماء تلقائياً: Completed, In Progress, At Risk
    autopct='%1.1f%%',
    startangle=140,
    colors=['#66b3ff', '#ffcc99', '#ff9999'],  # ألوان متناسقة لثلاثة أقسام
    wedgeprops={'edgecolor': 'black', 'linewidth': 1},
)
plt.title(
    'Student Status Distribution (3-States)', fontsize=14, fontweight='bold', pad=15
)
plt.savefig('charts/pie_chart_status_3parts.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# 2. رسم خريطة الارتباط الحرارية للمتغيرات الرقمية (Correlation Heatmap)
plt.figure(figsize=(8, 6))
corr_matrix = df_merged[
    ['Age', 'CompletionPercentage', 'Rating', 'Attempts', 'Score']
].corr()
sns.heatmap(
    corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5
)
plt.title(
    'Correlation Heatmap of Numerical Features', fontsize=14, fontweight='bold'
)
plt.tight_layout()
plt.savefig('charts/correlation_heatmap.png', dpi=300)
plt.show()
plt.close()

# 3. رسم التوزيع التكراري ومنحنى الكثافة لأعمار الطلاب (Histogram with KDE)
plt.figure(figsize=(9, 5))
sns.histplot(
    data=df_merged,
    x='Age',
    kde=True,
    color='purple',
    bins=20,
    edgecolor='black',
)
plt.title(
    'Age Distribution and Density of Students', fontsize=14, fontweight='bold'
)
plt.xlabel('Age', fontsize=12, fontweight='bold')
plt.ylabel('Count / Density', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/age_distribution_hist.png', dpi=300)
plt.show()
plt.close()

# 4. رسم المخطط الشريطي التراكمي لحالة الطلاب حسب الفئة العمرية (Stacked Bar Chart)
df_crosstab = pd.crosstab(df_merged['AgeGroup'], df_merged['StudentStatus'])
df_crosstab.plot(
    kind='bar',
    stacked=True,
    figsize=(10, 6),
    colormap='viridis',
    edgecolor='black',
)
plt.title(
    'Stacked Bar Chart of Student Status by Age Group',
    fontsize=14,
    fontweight='bold',
    pad=15,
)
plt.xlabel('Age Group', fontsize=12, fontweight='bold')
plt.ylabel('Number of Students', fontsize=12, fontweight='bold')
plt.xticks(rotation=0, fontweight='bold')
plt.legend(title='Student Status', title_fontsize='11', fontsize='10')
plt.tight_layout()
plt.savefig('charts/stacked_bar_chart_status.png', dpi=300)
plt.show()
plt.close()

# 5. رسم المخطط الخطي المزدوج لمعدل الرسوب والمحاولات حسب الدورة (Dual-Axis Line Plot)
course_stats_fail = (
    df_merged.groupby('CourseID')
    .agg(
        Failure_Rate=('Is_Failure', lambda x: x.mean() * 100),
        Avg_Attempts=('Attempts', 'mean'),
    )
    .reset_index()
)
top_10_failure = (
    course_stats_fail.sort_values(by='Failure_Rate', ascending=False)
    .head(10)
    .sort_values(by='Failure_Rate')
)

fig, ax1 = plt.subplots(figsize=(12, 6))
color = 'tab:red'
ax1.set_xlabel('Course ID', fontsize=12, fontweight='bold')
ax1.set_ylabel(
    'Failure Rate (%)', color=color, fontsize=12, fontweight='bold'
)
ax1.plot(
    top_10_failure['CourseID'],
    top_10_failure['Failure_Rate'],
    color=color,
    marker='o',
    linewidth=3,
    markersize=8,
)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel(
    'Average Attempts', color=color, fontsize=12, fontweight='bold'
)
ax2.plot(
    top_10_failure['CourseID'],
    top_10_failure['Avg_Attempts'],
    color=color,
    marker='s',
    linestyle='--',
    linewidth=3,
    markersize=8,
)
ax2.tick_params(axis='y', labelcolor=color)
plt.title(
    'Top 10 Courses with Highest Failure Rate & Average Attempts',
    fontsize=14,
    fontweight='bold',
    pad=15,
)
plt.tight_layout()
plt.savefig('charts/top_10_failure_courses.png', dpi=300)
plt.show()
plt.close()

# 6. رسم المخطط الشريطي لعدد الطلاب حسب سنة التسجيل (Bar Chart)
yearly_students = (
    df_merged.groupby('RegistrationYear')['StudentID'].nunique().reset_index()
)
plt.figure(figsize=(9, 5))
ax = sns.barplot(
    data=yearly_students,
    x='RegistrationYear',
    y='StudentID',
    palette='Set2',
    hue='RegistrationYear',
    legend=False,
)
plt.title(
    'Number of Students per Registration Year', fontsize=14, fontweight='bold'
)
plt.xlabel('Registration Year', fontsize=12, fontweight='bold')
plt.ylabel('Number of Students', fontsize=12, fontweight='bold')

# ---> أضف هذا السطر مع حلقة التكرار لإضافة الأرقام فوق الأعمدة <---
for p in ax.patches:
  height = p.get_height()
  if height > 0:
    ax.annotate(
        f'{int(height)}',
        (p.get_x() + p.get_width() / 2.0, height),
        ha='center',
        va='bottom',
        xytext=(0, 4),  # مسافة بسيطة فوق العمود
        textcoords='offset points',
        fontsize=11,
        fontweight='bold',
    )

plt.tight_layout()
plt.savefig('charts/students_per_year.png', dpi=300)
plt.show()
plt.close()


# 7. معالجة وحساب نسبة النجاح حسب الفئة العمرية ورسمها بشكل احترافي وواضح
success_by_age = (
    df_merged.groupby('AgeGroup', observed=False)['Is_Success']
    .mean()
    .reset_index()
)

# التأكد من تحويل القيمة إلى نسبة مئوية صحيحة (ضرب في 100 لتعرض بشكل صحيح من 0 إلى 100)
success_by_age['Success_Rate_Percentage'] = (
    success_by_age['Is_Success'] * 100
)

# إنشاء مساحة الرسم البياني وتحديد حجمها
plt.figure(figsize=(10, 5))

# رسم المخطط الشريطي (Bar Plot) لنسب النجاح لكل فئة عمرية
ax = sns.barplot(
    data=success_by_age,
    x='AgeGroup',
    y='Success_Rate_Percentage',
    palette='viridis',
    hue='AgeGroup',
    legend=False,
)

# إعداد عنوان المخطط وتسميات المحاور بخط عريض وواضح
plt.title(
    'Success Rate by Age Group (%)', fontsize=14, fontweight='bold', pad=15
)
plt.xlabel('Age Group', fontsize=12, fontweight='bold')
plt.ylabel('Success Rate (%)', fontsize=12, fontweight='bold')

# ضبط المحور الرأسي بناءً على النسب المئوية الحقيقية (حتى 100 أو أعلى قليلاً)
max_rate = success_by_age['Success_Rate_Percentage'].max()
plt.ylim(0, max_rate + 15)

# ضمان ظهور أسماء الفئات العمرية في منتصف الأعمدة تماماً وبخط عريض
plt.xticks(
    ticks=range(len(success_by_age)),
    labels=success_by_age['AgeGroup'],
    fontweight='bold',
)

# إضافة خطوط شبكية أفقية خفيفة لتحسين قراءة القيم ومقارنتها بدقة
plt.grid(axis='y', linestyle='--', alpha=0.5)

# حلقة تكرارية لمرور على كل عمود وكتابة النسبة المئوية الصحيحة بدقة فوقه مباشرة
for p in ax.patches:
  height = p.get_height()
  if height > 0:
    ax.annotate(
        f'{height:.1f}%',
        (p.get_x() + p.get_width() / 2.0, height),
        ha='center',
        va='bottom',
        xytext=(0, 5),
        textcoords='offset points',
        fontsize=11,
        fontweight='bold',
    )

# تحسين تنسيق الحواف وحفظ الرسم بدقة عالية (DPI 300) ثم عرضه وإغلاقه لتفريغ الذاكرة
plt.tight_layout()
plt.savefig('charts/success_rate_by_age_group.png', dpi=300)
plt.show()
plt.close()

# 8. رسم مخطط نسبة الإكمال حسب الدولة مع إظهار الأرقام بجانب الأشرطة الأفقية
country_stats = df_merged.groupby('Country')['Is_Completed'].mean().reset_index()
country_stats['Completion_Rate'] = (
    country_stats['Is_Completed'] * 100
)  # تحويل إلى نسبة مئوية
country_stats = country_stats.sort_values(
    by='Completion_Rate', ascending=True
)

plt.figure(figsize=(12, 7))
barplot = sns.barplot(
    data=country_stats,
    x='Completion_Rate',
    y='Country',
    palette='crest',
    hue='Country',
    legend=False,
)

plt.title(
    'Completion Rate by Country (%)', fontsize=16, fontweight='bold', pad=20
)
plt.xlabel('Completion Rate (%)', fontsize=12, fontweight='bold')
plt.ylabel('Country', fontsize=12, fontweight='bold')
plt.xlim(
    0, country_stats['Completion_Rate'].max() * 1.15
)  # توسيع المحور قليلاً لتسع الأرقام

# إضافة الأرقام بجانب الأشرطة الأفقية
for p in barplot.patches:
  width = p.get_width()
  if width > 0:
    barplot.annotate(
        f'{width:.1f}%',
        (width, p.get_y() + p.get_height() / 2.0),
        ha='left',
        va='center',
        xytext=(5, 0),
        textcoords='offset points',
        fontsize=11,
        fontweight='bold',
    )

plt.tight_layout()
plt.savefig('charts/completion_rate_by_country.png', dpi=300)
plt.show()
plt.close()

# 9. رسم المخطط الثنائي لمقارنة نسبة النجاح والمحاولات (Success & Attempts Bar Chart)
course_stats_perf = (
    df_merged.groupby('CourseID')
    .agg(
        Success_Rate=('Is_Success', lambda x: x.mean() * 100),
        Avg_Attempts=('Attempts', 'mean'),
    )
    .reset_index()
    .sort_values(by='Success_Rate', ascending=False)
    .head(10)
)

fig, ax1 = plt.subplots(figsize=(14, 6))
x = np.arange(len(course_stats_perf['CourseID']))
width = 0.35

rects1 = ax1.bar(
    x - width / 2,
    course_stats_perf['Success_Rate'],
    width,
    label='Success Rate (%)',
    color='tab:green',
)
ax1.set_ylabel(
    'Success Rate (%)', color='tab:green', fontsize=12, fontweight='bold'
)
ax1.set_xticks(x)
ax1.set_xticklabels(course_stats_perf['CourseID'], fontweight='bold')

ax2 = ax1.twinx()
rects2 = ax2.bar(
    x + width / 2,
    course_stats_perf['Avg_Attempts'],
    width,
    label='Average Attempts',
    color='tab:blue',
    alpha=0.8,
)
ax2.set_ylabel(
    'Average Attempts', color='tab:blue', fontsize=12, fontweight='bold'
)
plt.title(
    'Comparison of Success Rate and Average Attempts per Course',
    fontsize=14,
    fontweight='bold',
    pad=15,
)
fig.tight_layout()
plt.savefig('charts/success_attempts_barchart.png', dpi=300)
plt.show()
plt.close()

# 10. رسم مخطط الانحدار لدراسة العلاقة بين التقييم ونسبة الإكمال (Rating vs Completion Regplot)
merged_rev_enroll = pd.merge(
    reviews, enrollments, on=['StudentID', 'CourseID'], how='inner'
)
corr = merged_rev_enroll['Rating'].corr(
    merged_rev_enroll['CompletionPercentage']
)

plt.figure(figsize=(10, 6))
sns.regplot(
    data=merged_rev_enroll,
    x='Rating',
    y='CompletionPercentage',
    scatter_kws={'alpha': 0.3},
    line_kws={'color': 'red'},
)
plt.title(
    f'Correlation Rating vs Completion: {corr:.2f}',
    fontsize=14,
    fontweight='bold',
)
plt.savefig('charts/rating_vs_completion.png', dpi=300)
plt.show()
plt.close()

# 11. رسم مخطط الطلاب الذين يحتاجون متابعة حسب حالة التقدم (Students Needing Follow-up)
needs_followup = df_merged[
    (df_merged['CompletionPercentage'] < 30)
    | (df_merged['Progress'] == 'Not Started')
]
progress_counts = needs_followup['Progress'].value_counts().reset_index()

plt.figure(figsize=(9, 5))
sns.barplot(
    data=progress_counts,
    x='Progress',
    y='count',
    palette='Set2',
    hue='Progress',
    legend=False,
)
plt.title(
    'Students Needing Follow-up by Progress Status',
    fontsize=14,
    fontweight='bold',
)
plt.savefig('charts/students_needing_followup_chart.png', dpi=300)
plt.show()
plt.close()

# 12. رسم مخطط الدورات الأكثر نجاحاً والدورات الأكثر صعوبة وتعرّضاً للتعثر (Top & Hardest Courses)
course_perf = (
    df_merged.groupby('CourseID')
    .agg(
        Avg_Completion=('CompletionPercentage', 'mean'),
        Avg_Rating=('Rating', 'mean'),
    )
    .reset_index()
)

top_10_success = (
    course_perf.sort_values(by='Avg_Completion', ascending=False).head(10)
)
hardest_courses = (
    course_perf.sort_values(by='Avg_Completion', ascending=True).head(10)
)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=top_10_success,
    x='Avg_Completion',
    y='CourseID',
    palette='viridis',
    hue='CourseID',
    legend=False,
)
plt.title('Top 10 Most Successful Courses', fontsize=14, fontweight='bold')
plt.savefig('charts/top_successful_courses.png', dpi=300)
plt.show()
plt.close()

plt.figure(figsize=(10, 6))
sns.barplot(
    data=hardest_courses,
    x='Avg_Completion',
    y='CourseID',
    palette='Reds_r',
    hue='CourseID',
    legend=False,
)
plt.title(
    'Top 10 Courses with Highest Student Struggle',
    fontsize=14,
    fontweight='bold',
)
plt.savefig('charts/hardest_courses_struggle.png', dpi=300)
plt.show()
plt.close()

print('========================================')
print('All tasks, analyses, and 12 charts completed successfully!')
print('========================================')






def generate_executive_report():
  pipe = pipeline('text-generation', model='google/gemma-3-1b-it')

  # استخراج القيم الحقيقية كمتغيرات دقيقة لتجنب أي تخمين من النموذج
  top_age_group = success_by_age.iloc[0]['AgeGroup']
  top_age_rate = success_by_age.iloc[0]['Success_Rate_Percentage']

  top_country_name = success_by_country.iloc[0]['Country']
  top_country_rate = success_by_country.iloc[0]['Success_Rate_%']

  best_course = top_10_success.iloc[0]['CourseID']
  best_course_comp = top_10_success.iloc[0]['Avg_Completion']

  hardest_course = hardest_courses.iloc[0]['CourseID']
  hardest_course_comp = hardest_courses.iloc[0]['Avg_Completion']

  system_instruction = (
      'You are a professional strategic data analyst. CRITICAL RULE: You MUST'
      ' use the exact provided metrics without altering them. Do not invent'
      ' numbers, percentages, or statistics.'
  )

  user_query = f"""
    Write a professional executive report based strictly on these exact metrics:

    [Core Metrics]:
    - Total Unique Students: {total_students_unique}
    - Total Countries: {total_countries}
    - Average Student Age: {avg_age:.2f} years
    - Average Completion Percentage: {avg_completion:.2f}%
    - Overall Success Rate: {success_rate:.2f}%
    - Overall Failure Rate / At Risk: {failure_rate:.2f}%
    - Average Quiz Attempts: {avg_attempts:.2f}
    - Average Course Rating: {avg_rating:.2f} / 5

    [Key Findings to include directly]:
    - Top Performing Age Group: {top_age_group} with a success rate of {top_age_rate:.2f}%
    - Top Performing Country: {top_country_name} with a completion rate of {top_country_rate:.2f}%
    - Best Performing Course: Course {best_course} with an average completion of {best_course_comp:.2f}%
    - Most Difficult Course: Course {hardest_course} with an average completion of {hardest_course_comp:.2f}%

    Required Report Structure:
    1. Executive Summary & Objectives: Overview of the dataset, unique students ({total_students_unique}), countries ({total_countries}), and key performance indicators (KPIs).
    2. Data Methodology & Feature Engineering: Cleaning steps, missing values handling, and engineered features (AgeGroups: Under 20, 20-30, 31-40, 40+; StudentStatus: Completed, In Progress, At Risk; Is_Success, Is_Failure).
    3. Comprehensive Answers to Business Questions (Must explicitly and thoroughly answer using the exact numbers, age groups, and real CourseIDs from the tables above):
       1. Why do some students drop out? (Analyze dropout reasons and risk factors using the At Risk and failure metrics).
       2. Which demographics are the most successful? (Reference the exact AgeGroups and Country breakdowns provided above).
       3. Does age impact success? (Evaluate using the exact AgeGroup success rates).
       4. Does country affect completion rates? (Analyze geographical disparities using the country table).
       5. What is the correlation between attempts and success? (Examine quiz attempts and performance data).
       6. What is the relationship between ratings and completion percentage? (Analyze course ratings vs completion percentages).
       7. Which students require follow-up? (Define and identify the target group needing urgent follow-up based on status distribution).
       8. Which courses are the most successful? (List the actual CourseIDs from the Top Successful Courses table).
       9. Which courses do students struggle with the most? (List the actual CourseIDs from the Hardest Courses table).
       10. Five recommendations for management. (Provide structured, actionable strategic recommendations).
    4. Key Insights & Patterns: Deep statistical findings, engagement patterns, and course evaluations based strictly on the provided tables.
    5. Strategic Recommendations: Propose five practical, actionable, and structured recommendations for management to improve retention and learning outcomes.
    6. Conclusion: Summary and implementation roadmap.

    Please write the report in a professional corporate tone using ONLY the real numbers, exact age groups, and real CourseIDs provided above, avoiding any placeholders or fabricated data.
    """

  messages = [
      {'role': 'system', 'content': system_instruction},
      {'role': 'user', 'content': user_query},
  ]

  prompt = pipe.tokenizer.apply_chat_template(
      messages, tokenize=False, add_generation_prompt=True
  )

  response = pipe(prompt, max_new_tokens=2500)
  full_text = response[0]['generated_text']
  report_text = full_text.split('<start_of_turn>model')[-1].strip()
  return report_text

# استدعاء دالة التقرير وتوليده
final_report = generate_executive_report()
report_filename = 'Validated_Executive_Success_Report.md'

with open(report_filename, 'w', encoding='utf-8') as file:
  file.write(final_report)

print(
    f'\nValidated report successfully generated and exported to:'
    f' {report_filename}'
)


