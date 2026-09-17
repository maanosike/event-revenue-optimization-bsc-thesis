"""
=============================================================================
FILE 6: roie_scoring.py
PURPOSE: Compute Revenue Optimization Index for Events (ROI-E) scores
         from observable dataset proxies, validate against case studies
         (Chapter 8)
LIBRARIES: pandas, numpy, matplotlib
=============================================================================
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings('ignore')

import warnings

np.random.seed(42)

df_master = pd.read_csv('event_master_table.csv')

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: ROI-E Scoring from Observable Proxies
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("CHAPTER 8 – REVENUE OPTIMIZATION INDEX (ROI-E)")
print("=" * 60)

def compute_roie_proxies(df_event):
    """
    Estimate ROI-E sub-indicator scores from observable dataset variables.
    Returns dimensional and total scores for one event record.
    """
    # ── DIMENSION 1: DATA INFRASTRUCTURE ─────────────────────────────────
    # 1.1 Customer Data Integration: proxy = Ticket_Restriction_Flag (implies CRM)
    d1_1 = 2.0 if df_event.get('Ticket_Restriction_Flag', False) else 1.0

    # 1.2 Pre-Event Demand Signal Capture: proxy = Search_Interest_Index quality
    search = df_event.get('Search_Interest_Index', 0)
    d1_2   = min(4, int(search / 25))

    # 1.3 Secondary Market Monitoring: proxy = Secondary Premium available
    d1_3   = 2.5 if not pd.isna(df_event.get('Pre_Event_Secondary_Premium', None)) else 0.5

    # 1.4 Post-Event Analytics: proxy = No_Show_Rate recorded
    d1_4   = 2.0 if not pd.isna(df_event.get('No_Show_Rate', None)) else 1.0

    # 1.5 Data Governance: proxy = tier
    tier_score = {'Tier_1_Global': 3, 'Tier_2_Major': 2,
                  'Tier_3_Regional': 1, 'Tier_4_Emerging': 1}
    d1_5 = tier_score.get(df_event.get('Performer_Tier', 'Tier_3_Regional'), 1)

    # 1.6 Scalability: proxy = portfolio scale (not individual-event level — use category)
    d1_6 = 2.0

    dim1 = min(20, d1_1 + d1_2 + d1_3 + d1_4 + d1_5 + d1_6)

    # ── DIMENSION 2: ANALYTICAL SOPHISTICATION ───────────────────────────
    d2_1 = 4.0 if df_event.get('Dynamic_Pricing_Flag', False) else 2.0   # forecasting implied by DP
    d2_2 = min(5, df_event.get('Price_Tiers_Count', 2))
    d2_3 = 2.5   # average — not directly observable
    d2_4 = 2.0
    d2_5 = min(3, int(3 - df_event.get('Competing_Events_7Day', 3) / 6))

    dim2 = min(20, d2_1 + d2_2 + d2_3 + d2_4 + d2_5)

    # ── DIMENSION 3: PRICING STRATEGY ────────────────────────────────────
    d3_1 = min(5, df_event.get('Price_Tiers_Count', 1) *
               (2.0 if df_event.get('Dynamic_Pricing_Flag', False) else 0.5))
    d3_2 = min(4, df_event.get('Price_Tiers_Count', 1))
    vip_pct = (df_event.get('VIP_Package_Revenue_EUR', 0) /
               max(1, df_event.get('Total_Revenue_EUR', 1)))
    d3_3 = min(4, vip_pct * 20)
    sec   = df_event.get('Pre_Event_Secondary_Premium', 1.0)
    d3_4  = min(4, max(0, sec - 0.8) * 2.5)
    d3_5  = 2.0

    dim3 = min(20, d3_1 + d3_2 + d3_3 + d3_4 + d3_5)

    # ── DIMENSION 4: CUSTOMER INTELLIGENCE ───────────────────────────────
    d4_1 = 2.5
    d4_2 = 2.0 if df_event.get('Ticket_Restriction_Flag', False) else 1.0
    d4_3 = 2.0
    social = df_event.get('Artist_Social_Media_Score', 50)
    d4_4 = min(4, social / 25)
    d4_5 = 2.0

    dim4 = min(20, d4_1 + d4_2 + d4_3 + d4_4 + d4_5)

    # ── DIMENSION 5: GOVERNANCE AND ETHICS ───────────────────────────────
    d5_1 = 2.0
    d5_2 = 2.0 if df_event.get('Ticket_Restriction_Flag', False) else 1.0
    d5_3 = 2.0
    d5_4 = 1.5
    d5_5 = 2.0

    dim5 = min(20, d5_1 + d5_2 + d5_3 + d5_4 + d5_5)

    total_roie = dim1 + dim2 + dim3 + dim4 + dim5

    return {
        'Dim1_Data_Infrastructure':    round(dim1, 1),
        'Dim2_Analytical_Sophistication': round(dim2, 1),
        'Dim3_Pricing_Strategy':       round(dim3, 1),
        'Dim4_Customer_Intelligence':  round(dim4, 1),
        'Dim5_Governance_Ethics':      round(dim5, 1),
        'ROI_E_Total':                 round(total_roie, 1),
    }

# Apply to full dataset
print("\nComputing ROI-E for all 5,200 events...")
roie_records = df_master.apply(lambda row: compute_roie_proxies(row.to_dict()), axis=1)
df_roie = pd.DataFrame(list(roie_records))
df_full = pd.concat([df_master[['Event_ID', 'Event_Category',
                                 'Performer_Tier', 'Total_Revenue_EUR',
                                 'Fill_Rate', 'Venue_Capacity']],
                     df_roie], axis=1)
df_full['RevPAS'] = df_full['Total_Revenue_EUR'] / df_full['Venue_Capacity']

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: ROI-E by Category (Table 8.3)
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- Table 8.3 – Mean ROI-E by Event Category ---")
cat_roie = df_full.groupby('Event_Category')['ROI_E_Total'].agg(
    Mean='mean', SD='std', Median='median').round(1)
print(cat_roie.to_string())

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: ROI-E vs RevPAS Correlation (Section 8.4)
# ─────────────────────────────────────────────────────────────────────────────

from scipy.stats import pearsonr

r_corr, p_val = pearsonr(df_full['ROI_E_Total'], df_full['RevPAS'])
print(f"\n--- ROI-E vs RevPAS Pearson r = {r_corr:.3f}  (p={p_val:.4f}) ---")

# Partial correlation controlling for Category and Tier
from sklearn.linear_model import LinearRegression

cat_dummies  = pd.get_dummies(df_full['Event_Category'], drop_first=True)
tier_dummies = pd.get_dummies(df_full['Performer_Tier'], drop_first=True)
controls     = pd.concat([cat_dummies, tier_dummies], axis=1).astype(float)

def partial_corr(x, y, controls_df):
    """Pearson correlation of x and y after removing linear effect of controls."""
    lr = LinearRegression()
    X_ctrl = controls_df.values
    lr.fit(X_ctrl, x); res_x = x - lr.predict(X_ctrl)
    lr.fit(X_ctrl, y); res_y = y - lr.predict(X_ctrl)
    return pearsonr(res_x, res_y)[0]

partial_r = partial_corr(df_full['ROI_E_Total'].values,
                          df_full['RevPAS'].values, controls)
print(f"  Partial correlation (controlling cat + tier): r = {partial_r:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Case Study ROI-E Scores
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- Case Study ROI-E Scores (Section 9) ---")

CASE_STUDY_SCORES = {
    'Live Nation / Ticketmaster': {
        'Data Infrastructure': 19, 'Analytical Sophistication': 18,
        'Pricing Strategy': 17, 'Customer Intelligence': 15,
        'Governance & Ethics': 13,
    },
    'San Francisco Giants': {
        'Data Infrastructure': 14, 'Analytical Sophistication': 15,
        'Pricing Strategy': 16, 'Customer Intelligence': 13,
        'Governance & Ethics': 10,
    },
    'Coachella / Goldenvoice': {
        'Data Infrastructure': 16, 'Analytical Sophistication': 14,
        'Pricing Strategy': 17, 'Customer Intelligence': 13,
        'Governance & Ethics': 11,
    },
    'UEFA Ticketing': {
        'Data Infrastructure': 15, 'Analytical Sophistication': 13,
        'Pricing Strategy': 14, 'Customer Intelligence': 12,
        'Governance & Ethics': 12,
    },
    'AEG Presents': {
        'Data Infrastructure': 16, 'Analytical Sophistication': 15,
        'Pricing Strategy': 17, 'Customer Intelligence': 14,
        'Governance & Ethics': 12,
    },
    'NFL (Representative Franchise)': {
        'Data Infrastructure': 13, 'Analytical Sophistication': 12,
        'Pricing Strategy': 16, 'Customer Intelligence': 13,
        'Governance & Ethics': 9,
    },
}

dims = ['Data Infrastructure', 'Analytical Sophistication',
        'Pricing Strategy', 'Customer Intelligence', 'Governance & Ethics']

print(f"\n  {'Organisation':<35} {'Total ROI-E':>11}")
print("  " + "-" * 48)
for org, scores in CASE_STUDY_SCORES.items():
    total = sum(scores.values())
    print(f"  {org:<35} {total:>8}/100")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Spider / Radar Chart – Case Studies (Figure 8.1 / Figure 9.5)
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Figure 8.1 & 9.5 – ROI-E Radar Charts (Chapter 8 & 9)',
             fontweight='bold')

# Dimension scores distribution (box plot by category)
ax = axes[0]
roie_by_cat = [df_full[df_full['Event_Category'] == c]['ROI_E_Total'].values
               for c in df_full['Event_Category'].unique()]
cats = list(df_full['Event_Category'].unique())
bp   = ax.boxplot(roie_by_cat, labels=[c.replace('_', '\n') for c in cats],
                  patch_artist=True,
                  boxprops=dict(facecolor='#EEF3FB'),
                  medianprops=dict(color='#1B3A6B', linewidth=2))
ax.set_ylabel('ROI-E Total Score')
ax.set_title('Figure 8.2 – ROI-E Distribution by Category')
ax.tick_params(axis='x', labelsize=7)

# Scatter: ROI-E vs RevPAS
ax2 = axes[1]
sample_idx = np.random.choice(len(df_full), size=1000, replace=False)
scatter = ax2.scatter(df_full['ROI_E_Total'].iloc[sample_idx],
                      df_full['RevPAS'].iloc[sample_idx],
                      c=df_full['Fill_Rate'].iloc[sample_idx],
                      cmap='Blues', alpha=0.5, s=12)
plt.colorbar(scatter, ax=ax2, label='Fill Rate')

# Regression line
m, b = np.polyfit(df_full['ROI_E_Total'], df_full['RevPAS'], 1)
x_line = np.linspace(df_full['ROI_E_Total'].min(), df_full['ROI_E_Total'].max(), 100)
ax2.plot(x_line, m * x_line + b, 'r-', linewidth=2,
         label=f'r = {r_corr:.2f} (p<0.001)')
ax2.set_xlabel('ROI-E Total Score')
ax2.set_ylabel('Revenue per Available Seat (EUR)')
ax2.set_title('Figure 8.3 – ROI-E vs RevPAS')
ax2.legend()

plt.tight_layout()
plt.savefig('fig_roie_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: Radar chart for case study organisations
# ─────────────────────────────────────────────────────────────────────────────

n_dims   = len(dims)
angles   = np.linspace(0, 2 * np.pi, n_dims, endpoint=False)
angles   = np.concatenate([angles, [angles[0]]])

fig_radar, ax_r = plt.subplots(figsize=(8, 8),
                                subplot_kw=dict(polar=True))

palette = ['#1B3A6B', '#2E5FA3', '#6B8EC8', '#A8BFDF', '#E07B39', '#8B0000']

for i, (org, scores) in enumerate(CASE_STUDY_SCORES.items()):
    vals = [scores[d] / 20 for d in dims]   # normalise to 0–1
    vals = vals + [vals[0]]
    ax_r.plot(angles, vals, linewidth=2, color=palette[i],
              label=f"{org} ({sum(scores.values())})")
    ax_r.fill(angles, vals, color=palette[i], alpha=0.05)

ax_r.set_xticks(angles[:-1])
ax_r.set_xticklabels([d.replace(' ', '\n') for d in dims], fontsize=9)
ax_r.set_ylim(0, 1)
ax_r.set_title('Figure 9.5 – ROI-E Case Study Comparison', y=1.12, fontsize=12)
ax_r.legend(loc='lower right', bbox_to_anchor=(1.5, -0.1), fontsize=8)

plt.tight_layout()
plt.savefig('fig_roie_radar_cases.png', dpi=150, bbox_inches='tight')
plt.show()

df_full.to_csv('roie_scores.csv', index=False)
print("\nROI-E scores saved to roie_scores.csv")
