"""
=============================================================================
FILE 2: eda_feature_engineering.py
PURPOSE: Exploratory Data Analysis + Feature Engineering (Chapters 4 & 5)
LIBRARIES: pandas, numpy, matplotlib, seaborn, scipy
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from scipy import stats

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv('event_master_table.csv')
df_traj = pd.read_csv('event_trajectory_table.csv')

print("=" * 60)
print("CHAPTER 4 – EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Descriptive Statistics
# ─────────────────────────────────────────────────────────────────────────────

continuous_vars = [
    'Total_Revenue_EUR', 'Average_Ticket_Price_EUR', 'Fill_Rate',
    'Venue_Capacity', 'Pre_Event_Secondary_Premium',
    'Artist_Social_Media_Score', 'Search_Interest_Index',
    'Sales_Window_Days', 'VIP_Package_Revenue_EUR'
]

print("\n--- Descriptive Statistics (Table 4.7) ---")
desc = df[continuous_vars].describe(percentiles=[0.25, 0.50, 0.75])
desc.loc['skewness'] = df[continuous_vars].skew()
desc.loc['kurtosis'] = df[continuous_vars].kurt()
print(desc.round(2).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Event Distribution
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- Event Distribution by Category (Table 4.4) ---")
cat_dist = df.groupby('Event_Category').agg(
    Count=('Event_ID', 'count'),
    Mean_Fill_Rate=('Fill_Rate', 'mean'),
    Mean_Avg_Ticket=('Average_Ticket_Price_EUR', 'mean'),
    Pct_Dynamic_Pricing=('Dynamic_Pricing_Flag', 'mean'),
    Mean_Revenue=('Total_Revenue_EUR', 'mean'),
).round(3)
cat_dist['Count_Pct'] = (cat_dist['Count'] / len(df) * 100).round(1)
print(cat_dist.to_string())

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Dynamic Pricing Premium Analysis
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- Dynamic Pricing RevPAS Premium (Section 4.4.3) ---")
df['RevPAS'] = df['Total_Revenue_EUR'] / df['Venue_Capacity']

for cat in df['Event_Category'].unique():
    sub = df[df['Event_Category'] == cat]
    dp_rev  = sub[sub['Dynamic_Pricing_Flag'] == True]['RevPAS'].mean()
    std_rev = sub[sub['Dynamic_Pricing_Flag'] == False]['RevPAS'].mean()
    if std_rev > 0:
        uplift = (dp_rev / std_rev - 1) * 100
        t_stat, p_val = stats.ttest_ind(
            sub[sub['Dynamic_Pricing_Flag'] == True]['RevPAS'].dropna(),
            sub[sub['Dynamic_Pricing_Flag'] == False]['RevPAS'].dropna()
        )
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else "*")
        print(f"  {cat:<22}: DP uplift = +{uplift:.1f}%  {sig}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Pearson Correlation Matrix
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- Pearson Correlation Matrix (Table 4.11) ---")
corr_vars = [
    'Fill_Rate', 'Total_Revenue_EUR', 'Average_Ticket_Price_EUR',
    'Artist_Social_Media_Score', 'Search_Interest_Index',
    'Pre_Event_Secondary_Premium', 'Artist_Spotify_Listeners_M',
    'Competing_Events_7Day', 'Social_Sentiment_Score',
]
corr_matrix = df[corr_vars].corr()
print(corr_matrix.round(3).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Temporal Pattern Analysis
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- Temporal Sales Patterns (Section 4.6) ---")

# Announcement-day sales
ann_day = df_traj[df_traj['Week_Number'] == 1]
avg_ann_pct = (ann_day['Weekly_Net_Sales'] / df_traj.groupby('Event_ID')[
    'Weekly_Net_Sales'].transform('sum').loc[ann_day.index]).mean()
print(f"  Announcement-week mean sales share: {avg_ann_pct*100:.1f}%")

# 90-day cumulative
df_traj_90 = df_traj[df_traj['Weeks_To_Event'] >= 13]  # ~90 days
cum_90 = df_traj_90.groupby('Event_ID')['Weekly_Net_Sales'].sum()
total_sales = df_traj.groupby('Event_ID')['Weekly_Net_Sales'].sum()
pct_within_90 = (cum_90 / total_sales).mean()
print(f"  Mean % sales within 90 days of announcement: {pct_within_90*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: Visualisations
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Event Revenue Dataset – Exploratory Data Analysis (Chapter 4)',
             fontsize=14, fontweight='bold')

# Plot 1: Revenue distribution (log scale)
ax = axes[0, 0]
log_rev = np.log10(df['Total_Revenue_EUR'])
ax.hist(log_rev, bins=50, color='#1B3A6B', edgecolor='white', alpha=0.85)
ax.set_xlabel('log₁₀(Total Revenue EUR)')
ax.set_ylabel('Count')
ax.set_title('Figure 4.2 – Revenue Distribution (log scale)')
ax.axvline(log_rev.median(), color='orange', linestyle='--',
           label=f'Median: €{10**log_rev.median():,.0f}')
ax.legend(fontsize=8)

# Plot 2: Fill rate by category
ax = axes[0, 1]
fill_by_cat = [df[df['Event_Category'] == c]['Fill_Rate'].values
               for c in df['Event_Category'].unique()]
cats = df['Event_Category'].unique()
bp = ax.boxplot(fill_by_cat, labels=[c.replace('_', '\n') for c in cats],
                patch_artist=True,
                boxprops=dict(facecolor='#EEF3FB'),
                medianprops=dict(color='#1B3A6B', linewidth=2))
ax.set_ylabel('Fill Rate')
ax.set_title('Figure 4.3 – Fill Rate by Category')
ax.tick_params(axis='x', labelsize=6)

# Plot 3: Secondary premium vs performer tier
ax = axes[0, 2]
tier_order = ['Tier_1_Global', 'Tier_2_Major', 'Tier_3_Regional', 'Tier_4_Emerging']
colors = ['#1B3A6B', '#2E5FA3', '#6B8EC8', '#A8BFDF']
for i, tier in enumerate(tier_order):
    sub = df[df['Performer_Tier'] == tier]['Pre_Event_Secondary_Premium']
    ax.scatter(np.random.normal(i, 0.1, min(200, len(sub))),
               sub.sample(min(200, len(sub)), random_state=i),
               alpha=0.4, s=8, color=colors[i], label=tier.replace('_', ' '))
ax.set_xticks(range(4))
ax.set_xticklabels(['T1 Global', 'T2 Major', 'T3 Regional', 'T4 Emerging'],
                   fontsize=8)
ax.set_ylabel('Secondary Market Premium (×)')
ax.set_title('Figure 4.4 – Secondary Premium by Tier')
ax.axhline(1.0, color='red', linestyle='--', alpha=0.6, label='Face value')
ax.legend(fontsize=6)

# Plot 4: Correlation heatmap
ax = axes[1, 0]
short_names = ['Fill_Rate', 'Revenue', 'Avg_Price', 'Social',
               'Search', 'Sec_Prem', 'Spotify', 'Competing', 'Sentiment']
im = ax.imshow(corr_matrix.values, cmap='RdYlBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_vars)))
ax.set_yticks(range(len(corr_vars)))
ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=7)
ax.set_yticklabels(short_names, fontsize=7)
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title('Figure 4.5 – Pearson Correlation Heatmap')
for i in range(len(corr_vars)):
    for j in range(len(corr_vars)):
        val = corr_matrix.values[i, j]
        if abs(val) > 0.40:
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=6, color='white' if abs(val) > 0.6 else 'black')

# Plot 5: Average sales trajectory by tier
ax = axes[1, 1]
for tier, color in zip(tier_order, colors):
    tier_event_ids = df[df['Performer_Tier'] == tier]['Event_ID'].sample(
        min(100, len(df[df['Performer_Tier'] == tier])), random_state=42)
    traj_sub = df_traj[df_traj['Event_ID'].isin(tier_event_ids)]
    weekly_mean = traj_sub.groupby('Weeks_To_Event')['Cumulative_Fill_Rate'].mean()
    if len(weekly_mean) > 0:
        wk_sorted = weekly_mean.sort_index(ascending=False)
        ax.plot(range(len(wk_sorted)), wk_sorted.values,
                color=color, label=tier.replace('_', ' '), linewidth=1.5)
ax.set_xlabel('Weeks from Announcement')
ax.set_ylabel('Cumulative Fill Rate')
ax.set_title('Figure 4.6 – Sales Trajectory by Tier')
ax.legend(fontsize=7)

# Plot 6: Dynamic pricing premium by category
ax = axes[1, 2]
cat_dp   = df[df['Dynamic_Pricing_Flag'] == True].groupby(
    'Event_Category')['RevPAS'].mean()
cat_ndp  = df[df['Dynamic_Pricing_Flag'] == False].groupby(
    'Event_Category')['RevPAS'].mean()
cats_common = cat_dp.index.intersection(cat_ndp.index)
uplift_pct  = ((cat_dp[cats_common] / cat_ndp[cats_common]) - 1) * 100
bars = ax.barh(range(len(uplift_pct)),
               uplift_pct.values, color='#2E5FA3', edgecolor='#1B3A6B')
ax.set_yticks(range(len(uplift_pct)))
ax.set_yticklabels([c.replace('_', ' ')[:16] for c in cats_common], fontsize=8)
ax.set_xlabel('RevPAS Uplift vs Static Pricing (%)')
ax.set_title('Figure Dynamic Pricing Premium')
ax.axvline(0, color='black', linewidth=0.8)
for bar, val in zip(bars, uplift_pct.values):
    ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
            f'+{val:.1f}%', va='center', fontsize=7)

plt.tight_layout()
plt.savefig('fig_eda_chapter4.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigures saved to fig_eda_chapter4.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: Feature Engineering for ML Models (Chapter 5)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("CHAPTER 5 – FEATURE ENGINEERING")
print("=" * 60)

def engineer_features(df_traj, df_master):
    """Build the full 78-feature dataset for ML models."""
    df = df_traj.copy()

    # Merge master table
    master_cols = ['Event_ID', 'Event_Category', 'Performer_Tier', 'Market',
                   'Venue_Capacity', 'Average_Ticket_Price_EUR',
                   'Artist_Social_Media_Score', 'Artist_Spotify_Listeners_M',
                   'Pre_Event_Secondary_Premium', 'Competing_Events_7Day',
                   'Pandemic_Period', 'Dynamic_Pricing_Flag',
                   'Sales_Window_Days', 'Total_Revenue_EUR', 'Fill_Rate']
    df = df.merge(df_master[master_cols], on='Event_ID', how='left')

    # ── Temporal features ──────────────────────────────────────────────────
    df['Log_Weeks_To_Event']     = np.log1p(df['Weeks_To_Event'])
    df['Weeks_Since_Announce']   = df['Sales_Window_Days'] // 7 - df['Weeks_To_Event']
    df['Time_Fraction']          = 1 - (df['Weeks_To_Event'] /
                                       (df['Sales_Window_Days'] / 7 + 1))
    df['Is_Final_2_Weeks']       = (df['Weeks_To_Event'] <= 2).astype(int)
    df['Is_First_Week']          = (df['Week_Number'] == 1).astype(int)

    # ── Venue features ─────────────────────────────────────────────────────
    df['Log_Venue_Capacity']     = np.log(df['Venue_Capacity'])
    df['Capacity_Tier']          = pd.cut(df['Venue_Capacity'],
                                          bins=[0, 10000, 30000, 60000, 100000],
                                          labels=[0, 1, 2, 3]).astype(float)

    # ── Artist / performer features ────────────────────────────────────────
    df['Log_Spotify_Listeners']  = np.log1p(df['Artist_Spotify_Listeners_M'])
    df['Social_Score_Norm']      = df['Artist_Social_Media_Score'] / 100

    tier_enc = {'Tier_1_Global': 3, 'Tier_2_Major': 2,
                'Tier_3_Regional': 1, 'Tier_4_Emerging': 0}
    df['Tier_Numeric']           = df['Performer_Tier'].map(tier_enc)

    # ── Pricing features ───────────────────────────────────────────────────
    df['Price_Ratio']            = df['Current_Primary_Price'] / \
                                   (df['Average_Ticket_Price_EUR'] + 1e-6)
    df['Secondary_Premium']      = df['Secondary_Market_Price'] / \
                                   (df['Current_Primary_Price'] + 1e-6)

    # ── Demand signal features ─────────────────────────────────────────────
    # Exponential decay of search interest
    alpha = 0.15
    df = df.sort_values(['Event_ID', 'Week_Number'])
    df['Search_Decayed'] = df.groupby('Event_ID')['Weekly_Search_Index'].transform(
        lambda x: x.ewm(alpha=alpha, adjust=False).mean())

    df['Sentiment_MA7']          = df.groupby('Event_ID')['Weekly_Social_Sentiment']\
                                     .transform(lambda x: x.rolling(4, min_periods=1).mean())

    # ── Interaction features ───────────────────────────────────────────────
    df['Tier_x_LogCapacity']     = df['Tier_Numeric'] * df['Log_Venue_Capacity']
    df['PriceRatio_x_Social']    = df['Price_Ratio'] * df['Social_Score_Norm']
    df['Time_x_FillRate']        = df['Time_Fraction'] * df['Cumulative_Fill_Rate']
    df['Secondary_x_Tier']       = df['Secondary_Premium'] * df['Tier_Numeric']

    # ── One-hot encoding ───────────────────────────────────────────────────
    df = pd.get_dummies(df,
                        columns=['Event_Category', 'Market', 'Performer_Tier'],
                        drop_first=False, prefix=['cat', 'mkt', 'tier'])

    # ── Target variable ────────────────────────────────────────────────────
    df['Log_Weekly_Sales']       = np.log1p(df['Weekly_Net_Sales'])

    print(f"  Feature matrix shape: {df.shape}")
    print(f"  Non-null rate: {df.isnull().mean().mean()*100:.1f}% missing avg")
    return df

df_features = engineer_features(df_traj, df)
df_features.to_csv('event_features.csv', index=False)
print("  Saved: event_features.csv")
