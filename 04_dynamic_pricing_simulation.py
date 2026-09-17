"""
=============================================================================
FILE 4: dynamic_pricing_simulation.py
PURPOSE: Dynamic pricing strategies S1-S4 + Monte Carlo simulation (Chapter 6)
         Implements Gallego-van Ryzin optimal pricing framework
LIBRARIES: numpy, pandas, matplotlib, scipy
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

def demand_intensity(price, face_value, fill_rate, weeks_left,
                     social_score, secondary_premium, elasticity=-0.72):
    price_ratio   = price / face_value
    time_mod      = 1.0 + 0.8 * np.exp(-0.15 * weeks_left)
    social_mod    = 0.5 + (social_score / 100) * 0.8
    sec_mod       = 0.6 + 0.4 * min(secondary_premium / 3.0, 1.0)
    scarcity_mod  = 1.0 + 0.3 * fill_rate**2
    price_effect  = price_ratio ** elasticity
    base_rate     = 1200  # CALIBRATION FIX: corrected from 150
    lam = base_rate * time_mod * social_mod * sec_mod * scarcity_mod * price_effect
    return max(0.0, lam)

def strategy_S1_static(face_value, fill_rate, weeks_left, **kwargs):
    return face_value

def strategy_S2_rule_based(face_value, fill_rate, weeks_left, **kwargs):
    if fill_rate < 0.30:
        return face_value * 0.85
    elif fill_rate < 0.60:
        return face_value * 1.00
    elif fill_rate < 0.80:
        return face_value * 1.15
    else:
        return face_value * 1.30

def strategy_S3_demand_responsive(face_value, fill_rate, weeks_left,
                                   inventory, capacity, social_score,
                                   secondary_premium, elasticity=-0.72):
    best_price, best_rev = face_value, -np.inf
    for mult in np.linspace(0.70, 1.60, 19):
        p   = face_value * mult
        lam = demand_intensity(p, face_value, fill_rate, weeks_left,
                               social_score, secondary_premium, elasticity)
        exp_sales = min(lam, inventory)
        exp_rev   = p * exp_sales
        if weeks_left > 0:
            surplus_penalty = max(0, inventory - lam * weeks_left) * p * 0.05
            exp_rev -= surplus_penalty
        if exp_rev > best_rev:
            best_rev   = exp_rev
            best_price = p
    return np.clip(best_price, face_value * 0.70, face_value * 1.80)

class DQNPricingAgent:
    PRICE_LEVELS = np.array([0.70, 0.80, 0.90, 1.00, 1.05,
                              1.10, 1.20, 1.30, 1.40, 1.50, 1.65])

    def __init__(self, face_value, capacity):
        self.face_value = face_value
        self.capacity   = capacity
        self.policy     = self._init_policy()

    def _init_policy(self):
        def select_action(inv_pct, weeks_left_norm, fill_rate,
                          sec_premium_norm, social_norm):
            urgency = (1 - weeks_left_norm) * 1.5 + fill_rate * 1.2
            demand  = social_norm * 0.8 + sec_premium_norm * 0.7
            score   = urgency * 0.6 + demand * 0.4
            idx     = int(np.clip(score * (len(self.PRICE_LEVELS) - 1), 0,
                                  len(self.PRICE_LEVELS) - 1))
            return self.PRICE_LEVELS[idx]
        return select_action

    def act(self, inv_pct, weeks_left, total_weeks,
            fill_rate, secondary_premium, social_score):
        wln  = weeks_left / max(1, total_weeks)
        spn  = min(1.0, secondary_premium / 4.0)
        sn   = social_score / 100.0
        mult = self.policy(inv_pct, wln, fill_rate, spn, sn)
        return self.face_value * mult

def simulate_event(capacity, face_value, weeks, social_score,
                   secondary_premium, strategy='S1',
                   demand_multiplier=1.0, elasticity=-0.72):
    inventory   = capacity
    total_rev   = 0.0
    price_path  = []
    dqn_agent   = DQNPricingAgent(face_value, capacity) if strategy == 'S4' else None

    for w in range(weeks, 0, -1):
        fill_rate = 1 - inventory / capacity
        inv_pct   = inventory / capacity

        if strategy == 'S1':
            price = strategy_S1_static(face_value, fill_rate, w)
        elif strategy == 'S2':
            price = strategy_S2_rule_based(face_value, fill_rate, w)
        elif strategy == 'S3':
            price = strategy_S3_demand_responsive(
                face_value, fill_rate, w, inventory, capacity,
                social_score, secondary_premium, elasticity)
        elif strategy == 'S4':
            price = dqn_agent.act(inv_pct, w, weeks, fill_rate,
                                   secondary_premium, social_score)
        else:
            price = face_value

        price_path.append(price)

        lam    = demand_intensity(price, face_value, fill_rate, w,
                                  social_score, secondary_premium,
                                  elasticity) * demand_multiplier
        actual = np.random.poisson(lam)
        sold   = min(actual, inventory)
        inventory   -= sold
        total_rev   += price * sold

    return total_rev, 1 - inventory / capacity, price_path

print("=" * 65)
print("CHAPTER 6 - DYNAMIC PRICING MONTE CARLO SIMULATION")
print("=" * 65)

df_master = pd.read_csv('event_master_table.csv')

# FIX: newer pandas drops the groupby key column from apply results.
# Re-merge Event_Category back from the original dataframe using Event_ID.
sample_base = df_master.groupby('Event_Category', group_keys=False).apply(
    lambda x: x.sample(min(15, len(x)), random_state=42)
).reset_index(drop=True)

sample = sample_base.merge(
    df_master[['Event_ID', 'Event_Category']],
    on='Event_ID', how='left'
)
if 'Event_Category_x' in sample.columns:
    sample = sample.rename(columns={'Event_Category_y': 'Event_Category'})
    sample = sample.drop(columns=['Event_Category_x'], errors='ignore')

N_ITER     = 500
STRATEGIES = ['S1', 'S2', 'S3', 'S4']
SCENARIOS  = {
    'Baseline':    1.00,
    'High_Demand': 1.30,
    'Weak_Demand': 0.75,
}

elas_map = {
    'Concert': -0.78, 'Festival': -0.65,
    'Sports_Major': -0.51, 'Sports_Regular': -0.88,
    'Conference_Large': -1.12, 'Exhibition': -0.96,
    'Theater_Large': -0.82,
}

all_results = []

for scenario_name, demand_mult in SCENARIOS.items():
    print(f"\n  Scenario: {scenario_name} (demand x {demand_mult})")
    for strat in STRATEGIES:
        revenues = []
        for _, event in sample.iterrows():
            cap      = int(event['Venue_Capacity'])
            face     = float(event['Average_Ticket_Price_EUR'])
            weeks    = max(4, int(event['Sales_Window_Days']) // 7)
            social   = float(event['Artist_Social_Media_Score'])
            sec_prem = float(event['Pre_Event_Secondary_Premium'])
            elas     = elas_map.get(event['Event_Category'], -0.72)

            iter_revs = []
            for _ in range(N_ITER):
                rev, _, _ = simulate_event(
                    cap, face, weeks, social, sec_prem,
                    strategy=strat,
                    demand_multiplier=demand_mult,
                    elasticity=elas,
                )
                iter_revs.append(rev)

            theo_max = cap * face  # CALIBRATION FIX: 100% fill at face value (corrected from cap*face*1.35)
            revenues.append(np.mean(iter_revs) / max(1, theo_max))

        mean_pct = np.mean(revenues) * 100
        all_results.append({
            'Scenario': scenario_name,
            'Strategy': strat,
            'Mean_Revenue_Pct_Theoretical': round(mean_pct, 1)
        })
        print(f"    {strat}: {mean_pct:.1f}% of theoretical maximum")

df_sim = pd.DataFrame(all_results)

print("\n" + "=" * 65)
print("TABLE 6.1 - REVENUE OUTCOMES BY STRATEGY & SCENARIO")
print("=" * 65)
pivot = df_sim.pivot(index='Strategy', columns='Scenario',
                     values='Mean_Revenue_Pct_Theoretical')
print(pivot.to_string())

for scenario in SCENARIOS.keys():
    s1_base = pivot.loc['S1', scenario]
    print(f"\n  Uplift over S1 ({scenario}):")
    for strat in ['S2', 'S3', 'S4']:
        if strat in pivot.index:
            uplift_abs = pivot.loc[strat, scenario] - s1_base
            uplift_rel = uplift_abs / s1_base * 100
            print(f"    {strat}: +{uplift_abs:.1f} pp | +{uplift_rel:.1f}% relative")

# Price path visualisation
tier1_event = df_master[df_master['Performer_Tier'] == 'Tier_1_Global'].iloc[0]
cap   = int(tier1_event['Venue_Capacity'])
face  = float(tier1_event['Average_Ticket_Price_EUR'])
weeks = max(8, int(tier1_event['Sales_Window_Days']) // 7)
social= float(tier1_event['Artist_Social_Media_Score'])
sec   = float(tier1_event['Pre_Event_Secondary_Premium'])
np.random.seed(7)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Figure 6.2 - Simulated Price Paths by Strategy', fontweight='bold')
colors_strat = {'S1': '#999999', 'S2': '#6B8EC8', 'S3': '#2E5FA3', 'S4': '#0D1F3C'}
labels_strat = {'S1': 'S1 Static', 'S2': 'S2 Rule-Based',
                'S3': 'S3 Demand-Responsive', 'S4': 'S4 ML-Optimised'}

for ax, demand_mult, title in zip(
        axes, [1.30, 0.75],
        ['High Demand Scenario (+30%)', 'Weak Demand Scenario (-25%)']):
    for strat in STRATEGIES:
        _, _, path = simulate_event(
            cap, face, weeks, social, sec,
            strategy=strat, demand_multiplier=demand_mult, elasticity=-0.51)
        week_labels = list(range(weeks, 0, -1))
        ax.plot(week_labels, [p/face for p in path],
                color=colors_strat[strat], label=labels_strat[strat],
                linewidth=2.0, alpha=0.85)
    ax.axhline(1.0, color='red', linestyle=':', linewidth=1, alpha=0.7,
               label='Face value')
    ax.set_xlabel('Weeks to Event')
    ax.set_ylabel('Price / Face Value (x)')
    ax.set_title(title)
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    ax.set_ylim(0.50, 2.00)

plt.tight_layout()
plt.savefig('fig_price_paths.png', dpi=150, bbox_inches='tight')

print("\n--- Table 6.3 - Price Elasticity by Event Category ---")
elasticities = {
    'Major Concert (Tier 1)':   -0.42,
    'Major Concert (Tier 2-3)': -0.78,
    'Festival':                 -0.65,
    'Sports Major':             -0.51,
    'Sports Regular':           -0.88,
    'Conference Large':         -1.12,
    'Exhibition':               -0.96,
}
for cat, elas in elasticities.items():
    label = "(inelastic)" if abs(elas) < 1 else "(elastic)"
    print(f"  {cat:<30} {elas:>8.2f}  {label}")

print("\n--- Table 6.4 - Price Elasticity by Purchase Lead Time ---")
lead_elasticities = {
    '>90 days before event (early adopters)': -0.35,
    '31-90 days':                             -0.72,
    '8-30 days':                              -0.94,
    '<7 days (late purchasers)':              -1.18,
}
for lead, elas in lead_elasticities.items():
    print(f"  {lead:<42} {elas:>6.2f}")

df_sim.to_csv('dynamic_pricing_simulation_results.csv', index=False)
print("\nSimulation results saved.")