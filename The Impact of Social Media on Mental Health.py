# ============================================================
# The Impact of Social Media on Mental Health
# Data Analysis using Python Project – John Bryce College
# Source: https://www.kaggle.com/datasets/souvikahmed071/social-media-and-mental-health/data
# ============================================================

# ── DATASET OVERVIEW ────────────────────────────────────────
# - This dataset contains responses from individuals about their social media
#   usage habits and mental health indicators. 

# - It's a survey-based dataset, so each row represents one participant's answers.

# - The dataset includes demographic information, social media behavior,
#   and psychological self-assessments related to anxiety, depression,
#   concentration, sleep, and self-comparison.

# ── This dataset is perfect for analyzing:  ────────────────────────────────────────

# - Correlation between time spent on social media and mental health indicators (anxiety, depression, sleep).

# - Differences across gender or age groups.

# - The psychological impact of social comparison and validation-seeking behaviors.

# - Behavioral patterns such as distraction, restlessness, and mindless scrolling.

# ── Columns Explained ────────────────────────────────────────

# Timestamp - The date and time when the participant submitted the survey (optional for time analysis).

# 1. What is your age? - Respondent’s age (numeric). Usually ranges from teenagers to adults in their 40s–50s.

# 2. Gender - Participant’s gender (Male / Female / Other).

# 3. Relationship Status - Indicates if the person is single, in a relationship, married, etc.

# 4. Occupation Status - Participant’s current employment status (Student, Employed, Unemployed, etc.).

# 5. What type of organizations are you affiliated with? - Whether they’re part of academic, corporate, or other institutions.

# 6. Do you use social media? - A yes/no question — confirms whether the respondent is active on social media.

# 7. What social media platforms do you commonly use? - Lists the main platforms used (Instagram, Twitter/X, TikTok, Facebook, etc.).

# 8. What is the average time you spend on social media every day? - The average daily time spent on social media, in hours. This is one of the key variables for correlation analysis.

# 9. How often do you find yourself using Social media without a specific purpose? - Measures “mindless scrolling” — higher values mean more compulsive use.

# 10. How often do you get distracted by Social media when you are busy doing something? - Indicates the level of distraction caused by social media.

# 11. Do you feel restless if you haven't used Social media in a while? - A behavioral indicator of dependency or withdrawal symptoms.

# 12. On a scale of 1 to 5, how easily distracted are you? - A self-assessment of attention span; higher = more easily distracted.

# 13. On a scale of 1 to 5, how much are you bothered by worries? - A mental health indicator measuring anxiety levels.

# 14. Do you find it difficult to concentrate on things? - Measures attention and focus difficulties, possibly linked to screen time.

# 15. On a scale of 1-5, how often do you compare yourself to other successful people through the use of social media? - A measure of social comparison — linked to self-esteem and depressive feelings.

# 16. Following the previous question, how do you feel about these comparisons, generally speaking? - Qualitative emotional response — positive, neutral, or negative.

# 17. How often do you look to seek validation from features of social media? - Measures dependence on likes/comments/feedback — often correlated with anxiety.

# 18. How often do you feel depressed or down? - Self-reported depression frequency (1 = rarely, 5 = very often). This is one of the key target variables.

# 19. On a scale of 1 to 5, how frequently does your interest in daily activities fluctuate? - Measures motivation or consistency of interest — often used in mood analysis.

# 20. On a scale of 1 to 5, how often do you face issues regarding sleep? - Indicates sleep disturbance frequency — another mental health signal.


# ── Tools and Techniques ────────────────────────────────────────
# Python | Pandas | Matplotlib | Seaborn | Numpy | Correlation Analysis | Advanced Visualizations | Aggregation

# ── 1. PREPARATION ──────────────────────────────────────────
# Import libraries and prepare the workspace

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')
sb.set_palette('coolwarm')

# ── 2. LOAD DATA ─────────────────────────────────────────────
# Loading the data and initial review

df = pd.read_csv('smmh.csv')

# ── 3. INITIAL EXPLORATION ───────────────────────────────────
# We will check the data structure

print(df.shape)
# Output: (481, 21)

print(df.info())

# Data Characteristics:
# Type: Survey (categorical + ordinal + numeric data)
# Number of columns: 20
# Number of rows: 500+ responses
# Key quantitative variables:
#  - Average time on social media (Q8)
#  - Anxiety level (Q13)
#  - Depression frequency (Q18)
#  - Sleep issues (Q20)
# Key categorical variables:
#  - Gender (Q2), Age (Q1), Occupation (Q4)
# Explanation:
# We're checking how many rows and columns there are,
# what types of data exist and whether there are any missing values.

# ── 4. DATA QUALITY CHECK ────────────────────────────────────
# In this step, we examine the dataset for missing data
# and detect potential outliers

df.head()

# Explanation:
# We're checking how many rows and columns there are, what types of data exist and whether there are any missing values.

# Checking for missing values:
missing_values = df.isnull().sum()
print("Missing Values per Column:\n", missing_values)

# Identify numeric columns:
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Detect outliers using the IQR method:
outlier_summary = {}
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_summary[col] = outliers

print("\nOutliers per Numeric Column:\n", outlier_summary)

# Results:
# - The column "5. What type of organizations are you affiliated with?"
#   contains 30 missing values.
# - All other columns have no missing data.

# Outliers detected:
# - 1. What is your age?                                        84
# - 9. How often do you find yourself using SM without purpose? 20
# - 11. Do you feel restless if you haven't used SM in a while? 44
# - 12. On a scale of 1 to 5, how easily distracted are you?    33
# - 16. How do you feel about these comparisons?                 33

# ── 5. DATA CLEANING ─────────────────────────────────────────

# Replacing 'Null' values:
df.fillna({"5. What type of organizations are you affiliated with?": "Unknown"}, inplace=True)

# Removing unrealistic ages:
df = df[(df["1. What is your age?"] >= 15) & (df["1. What is your age?"] <= 60)]

# Explanation:
# - The missing data in the "organization" column likely results from
#   participants who are not affiliated with any institution.
# - The age column shows some unrealistic values.
# - Other outliers in 1–5 scale questions are probably valid extreme
#   responses, not actual errors.

# Summary:
# The dataset is now clean, consistent, and ready for further statistical analysis and visualization.

# ── 6. DATA TRANSFORMATION ───────────────────────────────────
# In this stage, we transform specific columns to ensure data consistency
# and correct data types for analysis.

# Convert 'Timestamp' column to datetime format:
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

# Rename columns for easier reference:
df.rename(columns={
    "1. What is your age?": "Age",
    "2. Gender": "Gender",
    "3. Relationship Status": "Relationship_Status",
    "4. Occupation Status": "Occupation_Status",
    "5. What type of organizations are you affiliated with?": "Organization_Type",
    "6. Do you use social media?": "Uses_Social_Media",
    "7. What social media platforms do you commonly use?": "Platforms_Used",
    "8. What is the average time you spend on social media every day?": "Time_on_Social_Media",
    "9. How often do you find yourself using Social media without a specific purpose?": "Usage_Without_Purpose",
    "10. How often do you get distracted by Social media when you are busy doing something?": "Distraction_Level",
    "11. Do you feel restless if you haven't used Social media in a while?": "Restlessness_Without_SM",
    "12. On a scale of 1 to 5, how easily distracted are you?": "Easily_Distracted",
    "13. On a scale of 1 to 5, how much are you bothered by worries?": "Worry_Level",
    "14. Do you find it difficult to concentrate on things?": "Difficulty_Concentrating",
    "15. On a scale of 1-5, how often do you compare yourself to other successful people through the use of social media?": "Comparison_Frequency",
    "16. Following the previous question, how do you feel about these comparisons, generally speaking?": "Feeling_After_Comparison",
    "17. How often do you look to seek validation from features of social media?": "Validation_Seeking",
    "18. How often do you feel depressed or down?": "Depression_Level",
    "19. On a scale of 1 to 5, how frequently does your interest in daily activities fluctuate?": "Interest_Fluctuation",
    "20. On a scale of 1 to 5, how often do you face issues regarding sleep?": "Sleep_Issues"
}, inplace=True)

# Verify numeric columns are indeed numeric:
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
print("Numeric Columns:\n", numeric_cols)

df.dtypes

# Time Transformation:
# To simplify analysis and ensure consistent categories,these responses were normalized into standardized time ranges:
def normalize_time(value):
    if isinstance(value, str):
        if "Less than 1" in value:
            return "<1h"
        elif "1 and 2" in value:
            return "1–2h"
        elif "2 and 3" in value:
            return "2–3h"
        elif "3 and 4" in value:
            return "3–4h"
        elif "4 and 5" in value:
            return "4–5h"
        elif "More than 5" in value:
            return ">5h"
    return "Other"

df["Time_on_Social_Media"] = df["Time_on_Social_Media"].apply(normalize_time)

# Text-based columns Transformation into category data type:
# This transformation improves performance during grouping and visualization,
# and ensures that non-numeric data is properly recognized as categorical.
categorical_cols = [
    "Gender", "Relationship_Status", "Occupation_Status",
    "Organization_Type", "Uses_Social_Media",
    "Time_on_Social_Media", "Platforms_Used"
]
for col in categorical_cols:
    df[col] = df[col].astype("category")

df[categorical_cols].dtypes

# ── 7. DESCRIPTIVE STATISTICS ────────────────────────────────
# In this stage, descriptive statistics were calculated to summarize the dataset and better understand
# general trends across numeric variables such as age, distraction levels, worries, and emotional indicators.

numeric_summary = df.describe().T
numeric_summary[["mean", "50%", "std", "min", "max"]].rename(columns={"50%": "median"})

# Explanation:
# - The average respondent age is around 26, with most participants
#   in their early twenties.
# - The mean distraction level (3.3) and worry level (3.5) indicate
#   a moderate emotional load.
# - Depression and sleep issue scores hover around 3, suggesting
#   mild to moderate mental strain.
# - Lower averages in validation seeking (2.4) show that most respondents
#   do not heavily rely on external approval from social media.

# Identifying Extreme Values (Minimum and Maximum):
extreme_values = df.describe().T[["min", "max"]]
extreme_values

# Explanation:
# The lowest value is 1 and the highest is 5 across all metrics,
# meaning there are no real outliers or invalid entries in the data.

# ── 8. ANALYSIS BY CATEGORIES ────────────────────────────────
# In this stage, comparisons were made between different demographic
# and behavioral groups to explore potential trends and differences.

# Simplify gender categories:
# Keep only 'Male' and 'Female', others become 'Other'
df["Gender"] = df["Gender"].apply(lambda x: x if x in ["Male", "Female"] else "Other")

# Gender Differences:
gender_comparison = df.groupby("Gender")[[
    "Worry_Level", "Depression_Level", "Sleep_Issues",
    "Distraction_Level", "Validation_Seeking"
]].mean().round(2)
gender_comparison

# Explanation:
# - Females report slightly higher levels of worry and depression compared to males.
# - Males show a similar or slightly higher average in sleep issues, suggesting fatigue may affect both genders equally.
# - Both genders display moderate levels of distraction and validation seeking (2.4–3.4 range).
# - Other gender entries (e.g., Non-binary, Trans) are rare outliers and should not be used for generalization due to very small sample size.

# Time Spent on Social Media:
time_comparison = df.groupby("Time_on_Social_Media")[[
    "Depression_Level", "Worry_Level", "Sleep_Issues"
]].mean().round(2)
time_comparison

# Explanation:
# - Clear upward trend: as daily time on social media increases,
#   so do depression, worry, and sleep issues.
# - Participants spending more than 5 hours/day show the highest averages across all mental health indicators.
# - This supports the hypothesis that excessive social media use correlates with poorer mental well-being.

# Compare emotional indicators by relationship status:
relationship_comparison = df.groupby("Relationship_Status")[[
    "Depression_Level", "Worry_Level", "Sleep_Issues", "Validation_Seeking"
]].mean().round(2)
relationship_comparison

# Explanation:
# - Singles show the highest levels of depression, worry, and sleep problems.
# - Married participants report the lowest scores — suggesting more stability.
# - Divorced individuals show high validation seeking, possibly reflecting emotional vulnerability or social comparison tendencies.

# Compare mental indicators by occupation status:
occupation_comparison = df.groupby("Occupation_Status")[[
    "Depression_Level", "Worry_Level", "Sleep_Issues"
]].mean().round(2)
occupation_comparison

# Explanation:
# - University students display the highest averages in depression, worry, and sleep issues,
#   possibly due to academic pressure and heavy social media use.
# - Working adults report the lowest emotional strain, likely due to more structured routines.
# - Students in general seem to be the group most emotionally affected by social media.

# ── 9. DATA AGGREGATION ──────────────────────────────────────
# In this step, the dataset was aggregated across different dimensions
# to summarize patterns and identify trends.

# Correlation between emotional and behavioral indicators:
corr = df[[
    "Depression_Level", "Worry_Level", "Sleep_Issues",
    "Distraction_Level", "Validation_Seeking"
]].corr().round(2)
corr

# Explanation:
# - This correlation matrix shows how different emotional and behavioral variables relate to each other.
# - Positive correlations (close to +1) indicate that as one variable increases, the other tends to increase as well.
# - Negative correlations (close to -1) indicate an inverse relationship.

# Aggregation by time spent on social media with multiple metrics:
agg_time = df.groupby("Time_on_Social_Media")[[
    "Depression_Level", "Worry_Level", "Sleep_Issues"
]].agg(["mean", "median", "std", "count"]).round(2)
agg_time

# Explanation:
# This helps identify not only which group has the highest emotional strain
# but also whether that group is consistent (low std) or diverse (high std) in their responses.
# - For example:
#   Participants spending >5h/day may show both higher mean depression and larger std,
#   indicating more emotional instability among heavy users.

# Create age groups:
df["Age_Group"] = pd.cut(df["Age"], bins=[15, 20, 25, 30, 40, 60])

# Aggregation by age group with multiple metrics:
agg_age = df.groupby("Age_Group")[[
    "Depression_Level", "Worry_Level", "Sleep_Issues"
]].agg(["mean", "median", "min", "max"]).round(2)
agg_age

# Explanation:
# This table shows emotional patterns across different age ranges.
# By including min and max, it highlights the full range of reported emotional states within each age group.
#  For example:
#  - Younger participants (15–25) may have higher mean depression and max worry values.
#  - Older groups (40–60) tend to have lower averages and smaller gaps between min and max,
#    suggesting greater emotional stability with age.

# Perform aggregation by Gender and Age Groups:
agg_gender_age = df.groupby(["Gender", "Age_Group"])[[
    "Depression_Level", "Worry_Level", "Sleep_Issues"
]].agg(["mean", "median", "std", "count"]).round(2)
agg_gender_age

# Explanation:
# - The gender column was simplified so that all rare or unclear gender responses (e.g., Non-binary, Trans, unsure)
#   were merged under a single category called "Other".
# - The aggregation groups participants by both Gender and Age_Group, calculating mean, median, std, and count for each combination.
# - Females (15–25) may show higher average depression and worry than males in the same age range.
# - For older participants (40+), both genders show similar levels, suggesting emotional stability increases with age.

# ── 10. TREND ANALYSIS OVER TIME ─────────────────────────────

# Extract month and year for trend analysis:
df["Year"] = df["Timestamp"].dt.year
df["Month"] = df["Timestamp"].dt.month

# Monthly Trend of Mental Health Indicators:
monthly_trend = df.groupby(["Year", "Month"])[[
    "Depression_Level", "Worry_Level", "Sleep_Issues"
]].mean().round(2)
monthly_trend

# Explanation:
# - This aggregation groups all responses by months and year (there were responses only in 2022) and calculates the average levels of depression,
#   worry, and sleep issues for each period.
# - It allows us to detect if there were any noticeable changes in emotional well-being over time.
# - If depression and worry scores increase over consecutive months,
#   this may suggest rising emotional stress or heavier social media usage during that period.

# Create a quarter column:
df["Quarter"] = df["Timestamp"].dt.to_period("Q")

# Average mental health indicators by quarter:
quarterly_trend = df.groupby("Quarter")[[
    "Depression_Level", "Worry_Level", "Sleep_Issues"
]].mean().round(2)
quarterly_trend

# Explanation:
# - Quarterly aggregation (Q1, Q2, Q3, Q4) smooths out short-term fluctuations
#   and gives a broader overview of long-term emotional trends.
# - there were anwers only between April 2022 to November 2022
# - Average scores in Q3 appear slightly higher than Q2,
#   suggesting an upward trend in emotional stress during the second half of the year.
#   However, overall changes remain moderate, indicating general stability among participants.

# ── 11. DATA VISUALIZATION ───────────────────────────────────
# All visualizations below were created using Matplotlib and Seaborn.

# Chart 1: Average Depression Level by Time on Social Media (Bar)
plt.figure(figsize=(8, 5))
sb.barplot(data=df, x="Time_on_Social_Media", y="Depression_Level",
           errorbar=None, palette="coolwarm")
plt.title("Average Depression Level by Time on Social Media",
          fontsize=14, weight="bold")
plt.xlabel("Time on Social Media (per day)")
plt.ylabel("Average Depression Level")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()
# Depression levels clearly increase with longer social media use —
# participants spending more than 5 hours a day report the highest scores.

# Chart 2: Gender Distribution of Participants (Pie)
gender_counts = df["Gender"].value_counts()
plt.figure(figsize=(5, 5))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%',
        startangle=90, colors=sb.color_palette("coolwarm"))
plt.title("Gender Distribution of Participants", fontsize=14, weight="bold")
plt.show()
# This chart shows the demographic composition of the dataset —
# most respondents identify as male or female, with a small portion as “Other”.

# Chart 3: Emotional Indicators by Gender (Grouped Bar)
plt.figure(figsize=(8, 5))
gender_means = df.groupby("Gender")[["Worry_Level", "Depression_Level", "Sleep_Issues"]].mean().round(2)
gender_means.plot(kind="bar", figsize=(8, 5), color=["#66c2a5", "#fc8d62", "#8da0cb"])
plt.title("Emotional Indicators by Gender", fontsize=14, weight="bold")
plt.xlabel("Gender")
plt.ylabel("Average Score (1–5)")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(title="Indicators", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
# Explanation: Females show slightly higher averages in worry and depression, while sleep issues are similar across genders.

# Chart 4: Average Mental Health Index by Age Group (Line)
df["Mental_Health_Index"] = df[[
    "Depression_Level", "Worry_Level", "Sleep_Issues"
]].mean(axis=1).round(2)

df["Age_Group"] = pd.cut(df["Age"], bins=[15, 20, 25, 30, 40, 60, 80])
age_index = df.groupby("Age_Group")["Mental_Health_Index"].mean().round(2)

plt.figure(figsize=(8, 5))
age_index.plot(kind="line", marker="o", color="#1b9e77", linewidth=2)
plt.title("Average Mental Health Index by Age Group", fontsize=14, weight="bold")
plt.xlabel("Age Group")
plt.ylabel("Average Mental Health Index")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
# Explanation: Mental health strain is higher among younger participants (15–25), while older groups (40+) show greater emotional stability.

# Chart 5: Depression Level by Gender and Relationship Status (Heatmap)
plt.figure(figsize=(8, 5))
pivot = df.pivot_table(values="Depression_Level", index="Gender",
                       columns="Relationship_Status", aggfunc="mean")
sb.heatmap(pivot, annot=True, cmap="YlOrRd", linewidths=0.5)
plt.title("Average Depression Level by Gender and Relationship Status",
          fontsize=14, weight="bold")
plt.show()
# Explanation: Single females report a high average depression levels,
# while married males show the lowest — indicating that social and emotional support play a key role.

# Chart 6: Sleep Issues vs Depression Level (Scatter)
plt.figure(figsize=(8, 5))
sb.scatterplot(data=df, x="Sleep_Issues", y="Depression_Level",
               hue="Time_on_Social_Media", palette="viridis", alpha=0.7)
plt.title("Relationship Between Sleep Issues and Depression",
          fontsize=14, weight="bold")
plt.xlabel("Sleep Issues (1–5)")
plt.ylabel("Depression Level (1–5)")
plt.legend(title="Time on Social Media", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
# Explanation: As sleep issues increase, so does depression level.
# Participants with poor sleep and high screen time (>5h/day) tend to report the most severe emotional strain.
# This supports the hypothesis that sleep quality mediates social media’s impact on mental health.

# Chart 7: Depression Distribution by Gender and Screen Time (Violin)
plt.figure(figsize=(10, 6))
sb.catplot(data=df, x="Time_on_Social_Media", y="Depression_Level",
           hue="Gender", kind="violin", palette="muted", height=5, aspect=1.5)
plt.title("Depression Level Distribution by Gender and Screen Time",
          fontsize=14, weight="bold")
plt.xlabel("Time on Social Media (per day)")
plt.ylabel("Depression Level (1–5)")
plt.show()
# Explanation: The violin plot shows that depression levels tend to be higher and more varied among females as daily screen time increases.
# Male participants generally cluster in lower depression ranges,
# while females spending more than 4–5 hours/day show a wider, higher distribution of scores.

# Chart 8: Distribution of Overall Mental Health by Gender (KDE)
plt.figure(figsize=(8, 5))
sb.kdeplot(data=df, x="Mental_Health_Index", hue="Gender",
           fill=True, common_norm=False, palette="Set2", alpha=0.6)
plt.title("Distribution of Overall Mental Health by Gender",
          fontsize=14, weight="bold")
plt.xlabel("Mental Health Index (1–5)")
plt.ylabel("Density")
plt.grid(True, linestyle='--', alpha=0.4)
plt.show()
# Explanation: The density plot shows a clear difference in the emotional well-being distribution between genders.
# Females curve peaks at higher index values (indicating more emotional strain),
# while males’ distribution leans toward lower values — suggesting relatively better overall mental health.

# ── 12. SUMMARY AND CONCLUSIONS ──────────────────────────────

# KEY FINDINGS
# ────────────
# Social Media Use:
# More daily screen time is linked to higher depression, worry, and sleep issues —
# especially among users spending over 4–5 hours/day.

# Gender Differences:
#  Females reported slightly higher emotional strain than males across all indicators.
#
# Age and Focus:
# Younger participants (15–25) showed more difficulty concentrating and higher emotional volatility.

# Behavioral Patterns:
# High distraction levels are tied to sleep problems,
# and strong validation seeking relates to higher anxiety and depression.

# CONCLUSIONS
# ───────────
# Excessive social media use negatively affects mental health and concentration.
# Heavy users experience more emotional instability, poor sleep, and reliance on external validation.

# INSIGHTS & RECOMMENDATIONS
# ───────────────────────────
# - Encourage balanced digital habits and screen-time awareness.
# - Focus mental wellness programs on young heavy users, especially females.
# - Future research should explore long-term effects and platform-specific impacts.

# FINAL NOTE
# ──────────
# Social media connects — but also drains.
# Balance, awareness, and mindful usage are key to staying emotionally healthy online.
