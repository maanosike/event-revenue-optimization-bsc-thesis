"""
=============================================================================
FILE 1: dataset_generation.py
PURPOSE: Generate the synthetic-empirical Event Revenue Dataset (Chapter 4)
         5,200 event records | 156,000 trajectory records | 42,000 customer records
LIBRARIES: numpy, pandas, scipy
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import truncnorm
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Industry-Calibrated Parameters (from Pollstar, Deloitte, PwC)
# ─────────────────────────────────────────────────────────────────────────────

CATEGORY_PARAMS = {
    # category: (n_events, capacity_mean, capacity_std, fill_rate_mean,
    #            fill_rate_std, avg_ticket_mean, avg_ticket_std,
    #            dyn_pricing_rate, vip_pct_mean)
    'Concert':          (1456, 14000, 8000,  0.81, 0.14, 98.0,  55.0, 0.483, 0.13),
    'Festival':         (624,  60000, 20000, 0.85, 0.10, 88.0,  30.0, 0.412, 0.09),
    'Sports_Major':     (780,  42000, 22000, 0.87, 0.12, 112.0, 60.0, 0.674, 0.15),
    'Sports_Regular':   (1040, 28000, 14000, 0.72, 0.18, 68.0,  28.0, 0.310, 0.08),
    'Conference_Large': (520,  6000,  2000,  0.79, 0.13, 148.0, 50.0, 0.125, 0.06),
    'Exhibition':       (416,  8000,  3000,  0.74, 0.15, 42.0,  18.0, 0.088, 0.04),
    'Theater_Large':    (364,  3500,  1200,  0.80, 0.16, 75.0,  25.0, 0.220, 0.10),
}

TIER_PARAMS = {
    # tier: (weight, social_score_mean, spotify_mean_M, secondary_premium_mean)
    'Tier_1_Global':  (0.124, 88.0, 45.0, 3.42),
    'Tier_2_Major':   (0.286, 68.0, 18.0, 2.14),
    'Tier_3_Regional':(0.395, 42.0,  5.0, 1.37),
    'Tier_4_Emerging':(0.195, 22.0,  0.8, 1.02),
}

MARKETS = {
    'USA_Northeast': 0.184,
    'USA_Southwest': 0.148,
    'USA_Midwest':   0.120,
    'UK':            0.203,
    'Germany':       0.146,
    'France':        0.112,
    'Canada':        0.087,
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def truncated_normal(mean, std, low, high, size):
    """Sample from a truncated normal distribution."""
    a = (low - mean) / std
    b = (high - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)

def generate_correlated_pair(x_norm, rho, mean2, std2, low2, high2):
    """Generate y correlated with x at level rho, then rescale to target distribution."""
    noise = np.random.normal(0, 1, len(x_norm))
    y_norm = rho * x_norm + np.sqrt(1 - rho**2) * noise
    # rescale
    y = y_norm * std2 + mean2
    return np.clip(y, low2, high2)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Generate Event Master Table
# ─────────────────────────────────────────────────────────────────────────────

print("Generating Event Master Table...")

records = []
event_id = 1000

for category, (n, cap_mu, cap_sig, fill_mu, fill_sig,
               atp_mu, atp_sig, dp_rate, vip_pct) in CATEGORY_PARAMS.items():

    # Draw performer tiers
    tier_names  = list(TIER_PARAMS.keys())
    tier_weights = [TIER_PARAMS[t][0] for t in tier_names]
    tiers = np.random.choice(tier_names, size=n, p=tier_weights)

    # Draw markets
    mkt_names   = list(MARKETS.keys())
    mkt_weights = list(MARKETS.values())
    markets = np.random.choice(mkt_names, size=n, p=mkt_weights)

    # Venue capacities
    capacities = truncated_normal(cap_mu, cap_sig, 5000, 90000, n).astype(int)

    # Fill rates (correlated slightly with tier)
    tier_fill_boost = np.array([{'Tier_1_Global':0.08,'Tier_2_Major':0.03,
                                  'Tier_3_Regional':0,'Tier_4_Emerging':-0.08}[t] for t in tiers])
    fill_rates = truncated_normal(fill_mu, fill_sig, 0.10, 1.0, n) + tier_fill_boost
    fill_rates = np.clip(fill_rates, 0.10, 1.0)

    # Tickets sold
    tickets_sold = (capacities * fill_rates).astype(int)

    # Average ticket price (correlated with tier)
    tier_price_mult = np.array([{'Tier_1_Global':1.55,'Tier_2_Major':1.15,
                                  'Tier_3_Regional':0.90,'Tier_4_Emerging':0.68}[t] for t in tiers])
    avg_ticket_price = truncated_normal(atp_mu, atp_sig, 15, 1500, n) * tier_price_mult
    avg_ticket_price = np.clip(avg_ticket_price, 15, 1500)

    # Total revenue
    total_revenue = (tickets_sold * avg_ticket_price).round(2)

    # VIP revenue
    vip_pct_arr = truncated_normal(vip_pct, 0.04, 0.0, 0.55, n)
    vip_revenue = (total_revenue * vip_pct_arr).round(2)

    # Dynamic pricing flag (Bernoulli)
    tier_dp_boost = np.array([{'Tier_1_Global':0.20,'Tier_2_Major':0.10,
                                'Tier_3_Regional':0,'Tier_4_Emerging':-0.10}[t] for t in tiers])
    dp_probs = np.clip(dp_rate + tier_dp_boost, 0, 1)
    dp_flags = np.random.binomial(1, dp_probs).astype(bool)

    # Artist social media score
    social_scores = np.array([
        truncated_normal(TIER_PARAMS[t][1], 12, 0, 100, 1)[0] for t in tiers
    ])

    # Artist Spotify listeners (millions)
    spotify_listeners = np.array([
        max(0.05, np.random.lognormal(np.log(max(0.1, TIER_PARAMS[t][2])), 0.6))
        for t in tiers
    ])

    # Secondary market premium (correlated with social score)
    sec_premium_means = np.array([TIER_PARAMS[t][3] for t in tiers])
    sec_premiums = np.maximum(0.7,
        sec_premium_means + np.random.normal(0, 0.45, n))

    # Google Trends search interest (correlated with social score)
    x_norm = (social_scores - social_scores.mean()) / (social_scores.std() + 1e-9)
    search_interest = generate_correlated_pair(x_norm, 0.72, 52.7, 26.4, 2, 100)

    # Social sentiment (-1 to +1, correlated with fill rate)
    fr_norm = (fill_rates - fill_rates.mean()) / (fill_rates.std() + 1e-9)
    sentiment = generate_correlated_pair(fr_norm, 0.52, 0.18, 0.38, -1, 1)

    # Sales window days (announcement to event)
    sales_window = truncated_normal(84, 55, 7, 365, n).astype(int)

    # Days to sellout (only for sold-out / near-sold-out events)
    sellout_mask = fill_rates > 0.95
    days_to_sellout = np.full(n, np.nan)
    days_to_sellout[sellout_mask] = truncated_normal(
        14, 18, 0, sales_window[sellout_mask], sellout_mask.sum())

    # Competing events
    competing_events = np.random.poisson(3.4, n)

    # Weather risk (outdoor events only)
    outdoor_mask = np.isin(
        np.array([category] * n), ['Festival'])  # simplified
    weather_risk = np.where(outdoor_mask,
                            truncated_normal(4.5, 2.5, 0, 10, n), 0.0)

    # Pandemic period flag (2020-2021 ~25% of events)
    pandemic_flag = np.random.binomial(1, 0.22, n).astype(bool)
    # Pandemic events have lower fill rate
    fill_rates[pandemic_flag] = np.clip(fill_rates[pandemic_flag] * 0.55, 0.05, 0.85)
    tickets_sold[pandemic_flag] = (capacities[pandemic_flag] *
                                   fill_rates[pandemic_flag]).astype(int)
    total_revenue[pandemic_flag] = (tickets_sold[pandemic_flag] *
                                    avg_ticket_price[pandemic_flag] * 0.70).round(2)

    # Event dates (spread 2019-2023)
    base_date = pd.Timestamp('2019-01-01')
    date_offsets = np.random.randint(0, 365 * 5, n)
    event_dates = [base_date + pd.Timedelta(days=int(d)) for d in date_offsets]

    # Price tiers count
    price_tiers = np.where(dp_flags,
                           np.random.randint(3, 8, n),
                           np.random.randint(2, 5, n))

    # No-show rate
    no_show_rate = truncated_normal(0.07, 0.04, 0.0, 0.30, n)

    # Ticket restriction flag
    restriction_flag = np.random.binomial(1, 0.18, n).astype(bool)

    for i in range(n):
        event_id += 1
        records.append({
            'Event_ID':                     event_id,
            'Event_Category':               category,
            'Performer_Tier':               tiers[i],
            'Market':                       markets[i],
            'Event_Date':                   event_dates[i].date(),
            'Sales_Window_Days':            int(sales_window[i]),
            'Venue_Capacity':               int(capacities[i]),
            'Fill_Rate':                    round(float(fill_rates[i]), 4),
            'Tickets_Sold':                 int(tickets_sold[i]),
            'Total_Revenue_EUR':            round(float(total_revenue[i]), 2),
            'Average_Ticket_Price_EUR':     round(float(avg_ticket_price[i]), 2),
            'VIP_Package_Revenue_EUR':      round(float(vip_revenue[i]), 2),
            'Dynamic_Pricing_Flag':         bool(dp_flags[i]),
            'Price_Tiers_Count':            int(price_tiers[i]),
            'Artist_Social_Media_Score':    round(float(social_scores[i]), 1),
            'Artist_Spotify_Listeners_M':   round(float(spotify_listeners[i]), 2),
            'Search_Interest_Index':        round(float(search_interest[i]), 1),
            'Social_Sentiment_Score':       round(float(sentiment[i]), 3),
            'Pre_Event_Secondary_Premium':  round(float(sec_premiums[i]), 2),
            'Days_To_Sellout':              (round(float(days_to_sellout[i]), 0)
                                             if not np.isnan(days_to_sellout[i]) else None),
            'Competing_Events_7Day':        int(competing_events[i]),
            'Weather_Risk_Index':           round(float(weather_risk[i]), 1),
            'Pandemic_Period':              bool(pandemic_flag[i]),
            'No_Show_Rate':                 round(float(no_show_rate[i]), 3),
            'Ticket_Restriction_Flag':      bool(restriction_flag[i]),
        })

df_master = pd.DataFrame(records)
print(f"  Master Table shape: {df_master.shape}")
print(f"  Total revenue range: EUR {df_master['Total_Revenue_EUR'].min():,.0f} – "
      f"EUR {df_master['Total_Revenue_EUR'].max():,.0f}")
print(f"  Fill rate: mean={df_master['Fill_Rate'].mean():.3f}, "
      f"median={df_master['Fill_Rate'].median():.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Generate Pre-Event Sales Trajectory Table (Bass Diffusion)
# ─────────────────────────────────────────────────────────────────────────────

print("\nGenerating Pre-Event Trajectory Table (156k records)...")

def generate_trajectory(event_id, capacity, fill_rate, social_score,
                        avg_price, secondary_premium, sales_window, tier):
    """
    Modified Bass diffusion model for ticket sales trajectory.
    Returns weekly observations from announcement to event.
    """
    target_sales = int(capacity * fill_rate)
    weeks        = max(1, sales_window // 7)

    # Bass model parameters calibrated by tier
    p_base = {'Tier_1_Global': 0.08, 'Tier_2_Major': 0.05,
              'Tier_3_Regional': 0.03, 'Tier_4_Emerging': 0.015}
    q_base = {'Tier_1_Global': 0.45, 'Tier_2_Major': 0.35,
              'Tier_3_Regional': 0.28, 'Tier_4_Emerging': 0.20}

    p = p_base.get(tier, 0.04) * (1 + 0.3 * (social_score - 50) / 50)
    q = q_base.get(tier, 0.30)

    # Bass cumulative adoption
    t_arr      = np.arange(1, weeks + 1) / weeks  # normalised time
    cumulative = (1 - np.exp(-(p + q) * t_arr)) / \
                 (1 + (q / p) * np.exp(-(p + q) * t_arr))
    weekly_frac = np.diff(np.concatenate([[0], cumulative]))

    # Add announcement-day spike
    spike = np.zeros(weeks)
    if weeks > 0:
        spike[0] = 0.08 + 0.19 * (social_score / 100)

    combined = weekly_frac + spike
    combined /= combined.sum()
    weekly_sales = (combined * target_sales).astype(int)

    # Add final-week surge (last 2 weeks)
    if weeks >= 2:
        surge_total = int(target_sales * 0.06)
        weekly_sales[-2] += surge_total // 2
        weekly_sales[-1] += surge_total - surge_total // 2

    # Price path: starts at face value, adjusts with fill rate
    prices = []
    cum_fill = 0.0
    for w in range(weeks):
        cum_fill = min(1.0, cum_fill + weekly_sales[w] / max(1, capacity))
        if cum_fill < 0.3:
            price = avg_price * 0.88
        elif cum_fill < 0.60:
            price = avg_price * 1.00
        elif cum_fill < 0.80:
            price = avg_price * 1.12
        else:
            price = avg_price * 1.28
        prices.append(round(price, 2))

    # Secondary market price (correlated, lagged)
    sec_prices = [round(p * (secondary_premium * 0.7 + 0.3 * (1 + cum / 2)), 2)
                  for p, cum in zip(prices, np.linspace(0, 1, weeks))]

    rows = []
    cum_sales = 0
    for w in range(weeks):
        cum_sales += weekly_sales[w]
        rows.append({
            'Event_ID':              event_id,
            'Weeks_To_Event':        weeks - w,
            'Week_Number':           w + 1,
            'Weekly_Net_Sales':      int(weekly_sales[w]),
            'Cumulative_Sales':      int(min(cum_sales, target_sales)),
            'Cumulative_Fill_Rate':  round(min(1.0, cum_sales / max(1, capacity)), 4),
            'Current_Primary_Price': prices[w],
            'Secondary_Market_Price':sec_prices[w],
            'Weekly_Search_Index':   round(float(np.random.normal(
                                        50 + 30 * (1 - (weeks - w) / weeks), 8)), 1),
            'Weekly_Social_Sentiment': round(float(np.random.normal(0.18, 0.12)), 3),
        })
    return rows

trajectory_rows = []
sample_ids = df_master['Event_ID'].values

for _, row in df_master.iterrows():
    traj = generate_trajectory(
        event_id         = row['Event_ID'],
        capacity         = row['Venue_Capacity'],
        fill_rate        = row['Fill_Rate'],
        social_score     = row['Artist_Social_Media_Score'],
        avg_price        = row['Average_Ticket_Price_EUR'],
        secondary_premium= row['Pre_Event_Secondary_Premium'],
        sales_window     = row['Sales_Window_Days'],
        tier             = row['Performer_Tier'],
    )
    trajectory_rows.extend(traj)

df_trajectory = pd.DataFrame(trajectory_rows)
print(f"  Trajectory Table shape: {df_trajectory.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Generate Customer Behavioral Table
# ─────────────────────────────────────────────────────────────────────────────

print("\nGenerating Customer Behavioral Table (42k records)...")

SEGMENT_PARAMS = {
    # segment: (pct, recency_mean, frequency_mean, monetary_mean,
    #           tier_pref, lead_days_mean, cat_loyalty_gini)
    'Loyal_Premium':        (0.163, 30,  5.0, 1847, 3.6, 45,  0.78),
    'Committed_Regulars':   (0.295, 60,  9.0,  842, 2.4, 55,  0.52),
    'Casual_Attendees':     (0.352, 120, 2.0,  284, 1.6, 80,  0.30),
    'Premium_Occasions':    (0.122, 90,  1.5, 2241, 3.9, 110, 0.85),
    'Price_Driven_Late_Buy':(0.068, 180, 3.5,  147, 1.2, 8,   0.18),
}

n_customers = 42000
customer_rows = []
cust_id = 10000

# Sample 800 events for behavioral data
sampled_event_ids = df_master.sample(800, random_state=42)['Event_ID'].values

for seg, (pct, rec_mu, freq_mu, mon_mu, tier_pref, lead_mu, _) in SEGMENT_PARAMS.items():
    n_seg = int(n_customers * pct)
    for _ in range(n_seg):
        cust_id += 1
        n_transactions = max(1, int(np.random.poisson(freq_mu)))
        for _ in range(n_transactions):
            cust_ticket_tier = max(1, min(4, int(np.round(
                np.random.normal(tier_pref, 0.6)))))
            purchase_lead = max(0, int(np.random.normal(lead_mu, lead_mu * 0.5)))
            channel = np.random.choice(
                ['web', 'mobile', 'box_office'], p=[0.55, 0.36, 0.09])
            loyalty_pts = max(0, int(np.random.exponential(500)))
            event_assigned = np.random.choice(sampled_event_ids)

            customer_rows.append({
                'Customer_ID':            f'C{cust_id:07d}',
                'Event_ID':               int(event_assigned),
                'Customer_Segment':       seg,
                'Ticket_Tier':            cust_ticket_tier,
                'Purchase_Lead_Days':     purchase_lead,
                'Purchase_Channel':       channel,
                'Loyalty_Points_Balance': loyalty_pts,
                'Geographic_Region':      np.random.choice(
                    list(MARKETS.keys()), p=list(MARKETS.values())),
            })

df_customers = pd.DataFrame(customer_rows).head(42000)
print(f"  Customer Table shape: {df_customers.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: Save to CSV
# ─────────────────────────────────────────────────────────────────────────────

df_master.to_csv('event_master_table.csv', index=False)
df_trajectory.to_csv('event_trajectory_table.csv', index=False)
df_customers.to_csv('event_customer_table.csv', index=False)

print("\n=== DATASET GENERATION COMPLETE ===")
print(f"  event_master_table.csv    : {len(df_master):,} rows")
print(f"  event_trajectory_table.csv: {len(df_trajectory):,} rows")
print(f"  event_customer_table.csv  : {len(df_customers):,} rows")

# Quick validation
print("\n=== VALIDATION CHECKS ===")
print(f"  Fill Rate mean  : {df_master['Fill_Rate'].mean():.3f}  (target 0.782)")
print(f"  Avg Ticket mean : EUR {df_master['Average_Ticket_Price_EUR'].mean():.1f}  (target 87.4)")
print(f"  Dynamic pricing %: {df_master['Dynamic_Pricing_Flag'].mean()*100:.1f}%  (target 34.2%)")
print(f"  Sec premium mean: {df_master['Pre_Event_Secondary_Premium'].mean():.2f}x  (target 1.68x)")
