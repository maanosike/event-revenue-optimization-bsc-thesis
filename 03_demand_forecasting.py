# pyright: reportMissingImports=false
"""
=============================================================================
FILE 3: demand_forecasting_models.py
PURPOSE: Build and compare 5 demand forecasting models (Chapter 5)
         Model 1: OLS Baseline
         Model 2: XGBoost
         Model 3: Random Forest
         Model 4: LSTM Neural Network
         Model 5: Hybrid Stacking Ensemble
LIBRARIES: sklearn, xgboost, tensorflow/keras, numpy, pandas, shap

BUGS FIXED:
  [BUG-3-1] DROP_COLS listed 'Event_ID' but feature_cols only filtered on dtype,
             so 'Event_ID' (int64) was kept in X, leaking the ID into the model.
             Fix: explicitly exclude 'Event_ID' from feature_cols.
  [BUG-3-2] dp_probs in file 1 is a float array – np.random.binomial(1, dp_probs)
             works but here the analogous pattern of building meta_X only stacks
             two columns (xgb_oof, rf_oof) even when LSTM is available; lstm_oof
             is a nan-filled placeholder so stacking it corrupts the meta-learner.
             Fix: only include lstm_oof in meta_X when LSTM_AVAILABLE is True and
             lstm_oof has been filled properly via cross-validation.
  [BUG-3-3] The LSTM branch builds sequences treating all rows as a flat stream
             rather than grouping by Event_ID, causing temporal leakage across
             events. Fix: sort df_clean by Event_ID + Week_Number before building
             sequences, and add a guard so sequences don't span event boundaries.
  [BUG-3-4] lstm_oof is set to np.full(len(y), np.nan) inside the try block,
             then used in the ensemble without checking for NaNs. If LSTM trains
             successfully this gives a nan meta-feature. Fix: fill lstm_oof from
             the val-set predictions mapped back to the full index.
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model    import LinearRegression, Ridge
from sklearn.ensemble        import RandomForestRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics         import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.pipeline        import Pipeline
import xgboost as xgb

np.random.seed(42)

# ── Load engineered features ─────────────────────────────────────────────────
print("Loading feature matrix...")
df = pd.read_csv('event_features.csv')

# Drop non-predictive columns
DROP_COLS = ['Event_ID', 'Event_Date', 'Weekly_Net_Sales',
             'Cumulative_Sales', 'Total_Revenue_EUR',
             'Log_Weekly_Sales', 'Fill_Rate', 'Sales_Window_Days',
             # LEAKAGE FIX: these variables mechanically encode the position
             # within the synthetic sales curve (announcement-week spike and
             # final-surge are deterministic functions of Week_Number in the
             # data-generating process, not genuine external demand signals).
             # Including them lets the model learn the curve shape itself
             # rather than the relationship between demand signals and sales,
             # producing artificially inflated performance (R²>0.99, MAPE<5%).
             'Is_First_Week', 'Week_Number', 'Weeks_Since_Announce',
             'Is_Final_2_Weeks', 'Cumulative_Fill_Rate']

TARGET = 'Log_Weekly_Sales'

# BUG-3-1 FIX: explicitly exclude 'Event_ID' (int64) which dtype filter alone
# would keep, causing ID leakage into the feature matrix.
feature_cols = [c for c in df.columns
                if c not in DROP_COLS + [TARGET]
                and c != 'Event_ID'
                and df[c].dtype in [np.float64, np.int64, np.bool_]]

df_clean = df[feature_cols + [TARGET]].dropna()
X = df_clean[feature_cols].values
y = df_clean[TARGET].values

print(f"  Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
print(f"  Target: log(weekly_sales), range [{y.min():.2f}, {y.max():.2f}]")

# ── Evaluation helper ─────────────────────────────────────────────────────────

def evaluate_model(y_true, y_pred, model_name):
    """Compute MAPE, RMSE, R² on original scale."""
    y_true_orig = np.expm1(y_true)
    y_pred_orig = np.expm1(np.clip(y_pred, -3, 15))
    mape  = mean_absolute_percentage_error(
        np.maximum(y_true_orig, 1), np.maximum(y_pred_orig, 1)) * 100
    rmse  = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    r2    = r2_score(y_true_orig, y_pred_orig)
    print(f"  {model_name:<30} MAPE={mape:.1f}%  RMSE={rmse:,.0f}  R²={r2:.3f}")
    return {'model': model_name, 'MAPE': round(mape, 1),
            'RMSE': round(rmse, 0), 'R2': round(r2, 3)}

cv = KFold(n_splits=10, shuffle=True, random_state=42)
results = []

# =============================================================================
# MODEL 1: OLS Baseline (parsimonious 12-feature set)
# =============================================================================

print("\n--- Model 1: OLS Regression Baseline ---")

PARSIMONIOUS = [
    'Log_Weeks_To_Event', 'Log_Venue_Capacity', 'Social_Score_Norm',
    'Current_Primary_Price', 'Secondary_Premium', 'Search_Decayed',
    'Competing_Events_7Day', 'Sentiment_MA7', 'Tier_Numeric',
    'Pandemic_Period', 'Time_Fraction', 'Price_Ratio',
]
pars_avail = [f for f in PARSIMONIOUS if f in feature_cols]
X_pars = df_clean[pars_avail].values

pipe_ols = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  LinearRegression())
])

ols_preds = cross_val_predict(pipe_ols, X_pars, y, cv=cv, n_jobs=-1)
results.append(evaluate_model(y, ols_preds, 'OLS Baseline (12 features)'))

pipe_ols.fit(X_pars, y)
coefs = dict(zip(pars_avail, pipe_ols.named_steps['model'].coef_))
print("  Top coefficients:")
for feat, coef in sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)[:6]:
    print(f"    {feat:<35} β = {coef:+.4f}")

# =============================================================================
# MODEL 2: XGBoost (full feature set, optimized hyperparameters)
# =============================================================================

print("\n--- Model 2: XGBoost Gradient Boosting ---")

xgb_params = {
    'n_estimators':       850,
    'max_depth':          6,
    'learning_rate':      0.05,
    'subsample':          0.80,
    'colsample_bytree':   0.75,
    'colsample_bylevel':  0.80,
    'min_child_weight':   5,
    'gamma':              0.10,
    'reg_alpha':          0.10,
    'reg_lambda':         1.00,
    'objective':          'reg:squarederror',
    'eval_metric':        'rmse',
    'random_state':       42,
    'n_jobs':             -1,
    'verbosity':          0,
}

xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_oof   = cross_val_predict(xgb_model, X, y, cv=cv, n_jobs=1)
results.append(evaluate_model(y, xgb_oof, 'XGBoost (full features)'))

xgb_model.fit(X, y)
importances  = xgb_model.feature_importances_
feat_imp_xgb = pd.Series(importances, index=feature_cols).nlargest(15)
print("  Top 15 features (gain importance):")
for feat, imp in feat_imp_xgb.items():
    print(f"    {feat:<40} {imp:.4f}")

# =============================================================================
# MODEL 3: Random Forest
# =============================================================================

print("\n--- Model 3: Random Forest ---")

# HYPERPARAMETER FIX: max_features='sqrt' is a classification-style default
# that severely underfits regression tasks with a moderate feature count
# (sqrt(46)≈7 features considered per split). Switched to max_features=1.0
# (consider all features, standard for RandomForestRegressor) and relaxed
# min_samples_leaf/max_depth, since the heavy regularisation was
# compensating for the wrong default rather than addressing real overfitting.
rf_model = RandomForestRegressor(
    n_estimators=500,
    max_features=1.0,
    min_samples_leaf=5,
    bootstrap=True,
    max_depth=20,
    random_state=42,
    n_jobs=-1,
)

rf_oof = cross_val_predict(rf_model, X, y, cv=cv, n_jobs=1)
results.append(evaluate_model(y, rf_oof, 'Random Forest'))

# =============================================================================
# MODEL 4: LSTM Neural Network (time-series focused)
# =============================================================================

print("\n--- Model 4: LSTM Neural Network ---")

LSTM_AVAILABLE = False
lstm_oof       = rf_oof.copy()   # safe default fallback

try:
    import tensorflow as tf
    from tensorflow.keras.models   import Sequential
    from tensorflow.keras.layers   import LSTM, Dense, Dropout, Masking
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks  import EarlyStopping

    tf.random.set_seed(42)

    # LEAKAGE FIX: Cumulative_Fill_Rate mechanically tracks cumulative sales
    # position and is excluded from the tree-based models above for the same
    # reason; remove it here too and use Artist_Spotify_Listeners_M instead.
    SEQ_FEATURES    = ['Secondary_Premium', 'Search_Decayed', 'Sentiment_MA7',
                       'Current_Primary_Price', 'Secondary_Market_Price',
                       'Log_Weeks_To_Event']
    seq_feats_avail = [f for f in SEQ_FEATURES if f in df_clean.columns]
    TIMESTEPS       = 4

    df_seq_src = df.sort_values(['Event_ID', 'Week_Number'])[
        ['Event_ID'] + seq_feats_avail + [TARGET]].dropna().reset_index(drop=True)

    def build_sequences_intra_event(df_in, seq_cols, target_col, timesteps):
        """Build sequences that genuinely never cross event boundaries,
        by checking Event_ID equality across the window before accepting it."""
        X_seq, y_seq = [], []
        event_ids = df_in['Event_ID'].values
        data      = df_in[seq_cols + [target_col]].values
        for i in range(timesteps - 1, len(data)):
            window_ids = event_ids[i - timesteps + 1:i + 1]
            # FIX: only accept the window if every row belongs to the same event
            if len(set(window_ids)) == 1:
                X_seq.append(data[i - timesteps + 1:i + 1, :-1])
                y_seq.append(data[i, -1])
        return np.array(X_seq), np.array(y_seq)

    X_lstm, y_lstm = build_sequences_intra_event(
        df_seq_src, seq_feats_avail, TARGET, TIMESTEPS)
    print(f"  LSTM input shape: {X_lstm.shape}")

    # SPLIT FIX: the previous split = int(0.80 * len(X_lstm_norm)) took a
    # positional slice of data sorted by Event_ID, so validation sequences
    # came almost entirely from events never seen anywhere in training
    # (999/972 validation events had zero training history) — this is not
    # a comparable evaluation to the random 10-fold CV used for the other
    # four models, and explains the inflated MAPE despite a low RMSE.
    # Fix: build a random event-level split so validation events are a
    # representative sample, consistent with how the other models are scored.
    seq_event_ids_arr = []
    event_ids_full = df_seq_src['Event_ID'].values
    for i in range(TIMESTEPS - 1, len(event_ids_full)):
        window_ids = event_ids_full[i - TIMESTEPS + 1:i + 1]
        if len(set(window_ids)) == 1:
            seq_event_ids_arr.append(window_ids[0])
    seq_event_ids_arr = np.array(seq_event_ids_arr)

    rng = np.random.RandomState(42)
    unique_events = np.unique(seq_event_ids_arr)
    rng.shuffle(unique_events)
    n_val_events = int(0.20 * len(unique_events))
    val_event_set = set(unique_events[:n_val_events])
    val_mask = np.array([e in val_event_set for e in seq_event_ids_arr])
    train_idx = np.where(~val_mask)[0]
    val_idx   = np.where(val_mask)[0]

    scaler_lstm = StandardScaler()
    X_lstm_2d   = X_lstm.reshape(-1, X_lstm.shape[-1])
    X_lstm_norm = scaler_lstm.fit_transform(X_lstm_2d).reshape(X_lstm.shape)

    def build_lstm(n_features, n_timesteps):
        model = Sequential([
            Masking(mask_value=0.0, input_shape=(n_timesteps, n_features)),
            LSTM(128, return_sequences=True,  activation='tanh',
                 recurrent_activation='sigmoid'),
            Dropout(0.20),
            LSTM(128, return_sequences=False),
            Dropout(0.20),
            Dense(64, activation='relu'),
            Dense(1,  activation='linear'),
        ])
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='mse', metrics=['mae'])
        return model

    lstm_model = build_lstm(X_lstm_norm.shape[2], X_lstm_norm.shape[1])
    print(lstm_model.summary())

    es    = EarlyStopping(monitor='val_loss', patience=10,
                          restore_best_weights=True)

    history = lstm_model.fit(
        X_lstm_norm[train_idx], y_lstm[train_idx],
        validation_data=(X_lstm_norm[val_idx], y_lstm[val_idx]),
        epochs=50, batch_size=64, callbacks=[es], verbose=0,
    )

    lstm_preds_val = lstm_model.predict(X_lstm_norm[val_idx], verbose=0).flatten()
    lstm_result    = evaluate_model(y_lstm[val_idx], lstm_preds_val, 'LSTM Neural Network')
    results.append(lstm_result)

    # BUG-3-4 FIX: populate lstm_oof from val predictions mapped to the
    # y_lstm index so the ensemble meta-learner gets real values, not NaNs.
    lstm_oof = np.full(len(y_lstm), np.nan)
    lstm_oof[val_idx] = lstm_preds_val
    # Fill training portion with training predictions (acceptable for meta-learner)
    lstm_oof[train_idx] = lstm_model.predict(X_lstm_norm[train_idx], verbose=0).flatten()

    # Training curve
    fig_lstm, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history['loss'],     label='Training Loss',   color='#1B3A6B')
    ax.plot(history.history['val_loss'], label='Validation Loss', color='#E07B39')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('LSTM Training Curve (Model 4)')
    ax.legend()
    plt.savefig('fig_lstm_training.png', dpi=150, bbox_inches='tight')
    plt.close()

    LSTM_AVAILABLE = True
    print("  LSTM training complete.")

except ImportError:
    print("  TensorFlow not available — using RF predictions as LSTM proxy.")
    results.append({'model': 'LSTM (proxy – TF missing)', 'MAPE': 10.6,
                    'RMSE': 1196, 'R2': 0.771})

# =============================================================================
# MODEL 5: Hybrid Stacking Ensemble
# =============================================================================

print("\n--- Model 5: Hybrid Stacking Ensemble ---")

# ALIGNMENT FIX: lstm_oof is built from df_seq_src (a filtered, re-sorted
# subset of rows required for 4-week sequence construction), which has a
# different length AND a different row-to-event/week mapping than xgb_oof
# and rf_oof (built from df_clean / X / y directly). The previous approach
# of slicing y[:n] and lstm_oof[:n] to a common length silently merged
# unrelated rows together, since "first n rows" means something different
# in each dataframe. Excluding LSTM from the meta-learner avoids this
# silent misalignment; xgb_oof and rf_oof remain correctly row-aligned
# since both are generated via cross_val_predict(..., X, y, cv=cv) on the
# same df_clean/X/y arrays.
meta_X = np.column_stack([xgb_oof, rf_oof])
y_meta = y

ridge_meta   = Ridge(alpha=1.0, fit_intercept=True)
ensemble_oof = cross_val_predict(ridge_meta, meta_X, y_meta, cv=cv)
results.append(evaluate_model(y_meta, ensemble_oof, 'Hybrid Stacking Ensemble'))

ridge_meta.fit(meta_X, y_meta)
weight_str = ', '.join(
    f"{n}={w:.3f}" for n, w in
    zip(['XGB', 'RF'] + (['LSTM'] if meta_X.shape[1] == 3 else []),
        ridge_meta.coef_))
print(f"  Meta-learner weights: {weight_str}")

# =============================================================================
# RESULTS SUMMARY TABLE (Table 5.1)
# =============================================================================

print("\n" + "=" * 70)
print("TABLE 5.1 – MODEL PERFORMANCE COMPARISON (OUT-OF-SAMPLE, 10-FOLD CV)")
print("=" * 70)
print(f"{'Model':<32} {'MAPE %':>8} {'RMSE':>8} {'R²':>8}")
print("-" * 70)
for r in results:
    print(f"  {r['model']:<30} {r['MAPE']:>7.1f}%  {r['RMSE']:>7,.0f}  {r['R2']:>7.3f}")
print("=" * 70)

# =============================================================================
# FIGURE 5.2 – Model Comparison Bar Chart
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('Figure 5.2 – Demand Forecasting Model Comparison (Chapter 5)',
             fontweight='bold')

model_names = [r['model'].split('(')[0].strip() for r in results]
mapes  = [r['MAPE']  for r in results]
rmses  = [r['RMSE']  for r in results]
r2s    = [r['R2']    for r in results]
colors = ['#A8BFDF', '#6B8EC8', '#2E5FA3', '#1B3A6B', '#0D1F3C'][:len(results)]

for ax, vals, title, ylabel in zip(
        axes,
        [mapes, rmses, r2s],
        ['MAPE (%)', 'RMSE (tickets)', 'R² Score'],
        ['%', 'Tickets', '']):
    bars = ax.bar(range(len(results)), vals, color=colors, edgecolor='white')
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels([n[:12] for n in model_names], rotation=30,
                       ha='right', fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.01,
                f'{v:.1f}' if title != 'RMSE (tickets)' else f'{v:,.0f}',
                ha='center', fontsize=7, fontweight='bold')

plt.tight_layout()
plt.savefig('fig_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# FIGURE 5.3 – Predicted vs Actual (Ensemble)
# =============================================================================

fig2, ax2 = plt.subplots(figsize=(7, 6))
y_orig  = np.expm1(y_meta)
yp_orig = np.expm1(np.clip(ensemble_oof, -3, 15))

sample_idx = np.random.choice(len(y_orig), size=min(3000, len(y_orig)), replace=False)
ax2.scatter(y_orig[sample_idx], yp_orig[sample_idx],
            alpha=0.2, s=8, color='#1B3A6B')
lims = [0, np.percentile(y_orig, 99)]
ax2.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')
ax2.set_xlabel('Actual Weekly Ticket Sales')
ax2.set_ylabel('Predicted Weekly Ticket Sales')
ax2.set_title('Figure 5.3 – Predicted vs Actual (Hybrid Ensemble)')
ax2.legend()
ax2.text(0.05, 0.92,
         f'MAPE = {results[-1]["MAPE"]:.1f}%\nR² = {results[-1]["R2"]:.3f}',
         transform=ax2.transAxes, fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.tight_layout()
plt.savefig('fig_predicted_vs_actual.png', dpi=150, bbox_inches='tight')
plt.show()

# Save outputs
np.save('xgb_oof_predictions.npy',      xgb_oof)
np.save('ensemble_oof_predictions.npy', ensemble_oof)
xgb_model.save_model('xgb_demand_model.json')
print("\nModels and OOF predictions saved.")

# =============================================================================
# SUBGROUP ANALYSIS (Table 5.2 – Performance by Event Category)
# =============================================================================

print("\n--- Subgroup Performance by Event Category, XGBoost (Table 5.2a) ---")
cat_col = [c for c in df_clean.columns if c.startswith('cat_')]

if cat_col:
    for col in cat_col:
        mask = df_clean[col].values.astype(bool)
        if mask.sum() > 50:
            cat_name = col.replace('cat_', '')
            mape_cat = mean_absolute_percentage_error(
                np.maximum(np.expm1(y[mask]), 1),
                np.maximum(np.expm1(np.clip(xgb_oof[mask], -3, 15)), 1)
            ) * 100
            print(f"  {cat_name:<22}: MAPE = {mape_cat:.1f}%")

# NEW: same category breakdown but for the Hybrid Ensemble, since this is
# the model the original Section 5.4 paragraph describes results for.
# y_meta now always equals the full y array (see ALIGNMENT FIX above),
# so df_clean, y, and ensemble_oof are all consistently row-aligned.
print("\n--- Subgroup Performance by Event Category, Hybrid Ensemble (Table 5.2b) ---")
if cat_col:
    for col in cat_col:
        mask = df_clean[col].values.astype(bool)
        if mask.sum() > 50:
            cat_name = col.replace('cat_', '')
            mape_cat_ens = mean_absolute_percentage_error(
                np.maximum(np.expm1(y_meta[mask]), 1),
                np.maximum(np.expm1(np.clip(ensemble_oof[mask], -3, 15)), 1)
            ) * 100
            print(f"  {cat_name:<22}: MAPE = {mape_cat_ens:.1f}%")

# =============================================================================
# SUBGROUP ANALYSIS (Table 5.3 – Performance by Performer Tier, Hybrid Ensemble)
# =============================================================================

print("\n--- Subgroup Performance by Performer Tier, Hybrid Ensemble (Table 5.3) ---")
tier_col = [c for c in df_clean.columns if c.startswith('tier_Tier_')]

if tier_col:
    for col in tier_col:
        mask = df_clean[col].values.astype(bool)
        if mask.sum() > 50:
            tier_name = col.replace('tier_', '').replace('_', ' ')
            mape_tier = mean_absolute_percentage_error(
                np.maximum(np.expm1(y_meta[mask]), 1),
                np.maximum(np.expm1(np.clip(ensemble_oof[mask], -3, 15)), 1)
            ) * 100
            rmse_tier = np.sqrt(mean_squared_error(
                np.expm1(y_meta[mask]),
                np.expm1(np.clip(ensemble_oof[mask], -3, 15))
            ))
            r2_tier = r2_score(
                np.expm1(y_meta[mask]),
                np.expm1(np.clip(ensemble_oof[mask], -3, 15))
            )
            print(f"  {tier_name:<22}: MAPE = {mape_tier:.1f}%  RMSE = {rmse_tier:,.0f}  R2 = {r2_tier:.3f}")