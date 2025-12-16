# =============================================================================
# 3. EVALUATE MODELS
# =============================================================================

print("="*70)
print("EVALUATING MODELS")
print("="*70)

results = []

# --- Model 1: Linear Regression ---
print("\n📊 Linear Regression...")
try:
    model = load('models/linear_regression_model.joblib')
    y_pred = model.predict(X_test)
    
    results.append({
        'Model': 'Linear Regression',
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred),
        'MAPE': np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100,
        'predictions': y_pred,
        'actual': y_test.values
    })
    print(f"   ✅ MAE: {results[-1]['MAE']:.2f} | RMSE: {results[-1]['RMSE']:.2f} | R²: {results[-1]['R2']:.4f}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# --- Model 2: LightGBM ---
print("\n📊 LightGBM...")
try:
    model = load('models/lgbm_model.joblib')
    y_pred = model.predict(X_test)
    
    results.append({
        'Model': 'LightGBM',
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred),
        'MAPE': np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100,
        'predictions': y_pred,
        'actual': y_test.values
    })
    print(f"   ✅ MAE: {results[-1]['MAE']:.2f} | RMSE: {results[-1]['RMSE']:.2f} | R²: {results[-1]['R2']:.4f}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# --- Model 3: SARIMA ---
print("\n📊 SARIMA...")
try:
    # Load SARIMA model
    with open('models/sarima_model.pkl', 'rb') as f:
        sarima_model = pickle.load(f)
    
    # Load test data
    test_ts = pd.read_csv('models/sarima_test_data.csv', index_col=0, parse_dates=True)
    
    # Forecast
    sarima_forecast = sarima_model.forecast(steps=len(test_ts))
    y_test_sarima = test_ts['units_sold'].values
    
    results.append({
        'Model': 'SARIMA',
        'MAE': mean_absolute_error(y_test_sarima, sarima_forecast),
        'RMSE': np.sqrt(mean_squared_error(y_test_sarima, sarima_forecast)),
        'R2': r2_score(y_test_sarima, sarima_forecast),
        'MAPE': np.mean(np.abs((y_test_sarima - sarima_forecast) / (y_test_sarima + 1e-10))) * 100,
        'predictions': sarima_forecast,
        'actual': y_test_sarima
    })
    print(f"   ✅ MAE: {results[-1]['MAE']:.2f} | RMSE: {results[-1]['RMSE']:.2f} | R²: {results[-1]['R2']:.4f}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# =============================================================================
# 4. COMPARISON RESULTS
# =============================================================================

if len(results) == 0:
    print("\n❌ No models evaluated! Run 'train_models.py' first.")
else:
    print("\n" + "="*70)
    print("COMPARISON TABLE")
    print("="*70)
    
    # Create comparison dataframe
    df_comp = pd.DataFrame([{k: v for k, v in r.items() if k not in ['predictions', 'actual']} 
                            for r in results])
    df_comp = df_comp.sort_values('RMSE')
    
    print(df_comp.to_string(index=False))
    
    # Best model
    best = df_comp.iloc[0]
    print("\n" + "="*70)
    print("🏆 BEST MODEL")
    print("="*70)
    print(f"Model : {best['Model']}")
    print(f"MAE   : {best['MAE']:.2f} unit")
    print(f"RMSE  : {best['RMSE']:.2f} unit")
    print(f"R²    : {best['R2']:.4f}")
    print(f"MAPE  : {best['MAPE']:.2f}%")
    
    # Interpretation
    if best['R2'] > 0.9:
        status = "⭐ EXCELLENT - Model sangat akurat!"
    elif best['R2'] > 0.7:
        status = "✅ GOOD - Model cukup akurat"
    elif best['R2'] > 0.5:
        status = "⚠️ FAIR - Model lumayan"
    else:
        status = "❌ POOR - Model perlu improvement"
    
    print(f"\n{status}")
    print(f"\n💡 Artinya: Model ini rata-rata meleset {best['MAE']:.0f} unit dari nilai aktual.")
    
    # =============================================================================
    # 5. VISUALIZATION
    # =============================================================================
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Model Evaluation Results - ML vs Time Series', fontsize=16, fontweight='bold', y=0.995)
    

    # Plot 1: RMSE Comparison
    ax = axes[0, 0]
    colors = ['#2ecc71' if i == 0 else '#e74c3c' for i in range(len(df_comp))]
    bars = ax.barh(df_comp['Model'], df_comp['RMSE'], color=colors, alpha=0.8)
    ax.set_xlabel('RMSE (Lower = Better)', fontweight='bold')
    ax.set_title('RMSE Comparison', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    for bar, val in zip(bars, df_comp['RMSE']):
        ax.text(val, bar.get_y() + bar.get_height()/2, f' {val:.2f}', 
                va='center', fontweight='bold')
    
    # Plot 2: R² Score Comparison
    ax = axes[0, 1]
    colors = ['#2ecc71' if df_comp.iloc[i]['R2'] == df_comp['R2'].max() 
              else '#e74c3c' for i in range(len(df_comp))]
    bars = ax.barh(df_comp['Model'], df_comp['R2'], color=colors, alpha=0.8)
    ax.set_xlabel('R² Score (Higher = Better)', fontweight='bold')
    ax.set_title('R² Score Comparison', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    for bar, val in zip(bars, df_comp['R2']):
        ax.text(val, bar.get_y() + bar.get_height()/2, f' {val:.4f}', 
                va='center', fontweight='bold')
    
    # Plot 3: MAPE Comparison
    ax = axes[0, 2]
    colors = ['#2ecc71' if i == 0 else '#e74c3c' for i in range(len(df_comp))]
    bars = ax.barh(df_comp['Model'], df_comp['MAPE'], color=colors, alpha=0.8)
    ax.set_xlabel('MAPE % (Lower = Better)', fontweight='bold')
    ax.set_title('MAPE Comparison', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    for bar, val in zip(bars, df_comp['MAPE']):
        ax.text(val, bar.get_y() + bar.get_height()/2, f' {val:.1f}%', 
                va='center', fontweight='bold')
    
    # Plot 4: Actual vs Predicted (Best Model)
    ax = axes[1, 0]
    best_result = [r for r in results if r['Model'] == best['Model']][0]
    best_pred = best_result['predictions']
    best_actual = best_result['actual']
    
    ax.scatter(best_actual, best_pred, alpha=0.5, s=15, color='#3498db')
    ax.plot([best_actual.min(), best_actual.max()], 
            [best_actual.min(), best_actual.max()], 
            'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Demand', fontweight='bold')
    ax.set_ylabel('Predicted Demand', fontweight='bold')
    ax.set_title(f'{best["Model"]} - Actual vs Predicted', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Residual Plot
    ax = axes[1, 1]
    residuals = best_actual - best_pred
    ax.scatter(best_pred, residuals, alpha=0.5, s=15, color='#3498db')
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Predicted Demand', fontweight='bold')
    ax.set_ylabel('Residuals (Actual - Predicted)', fontweight='bold')
    ax.set_title('Residual Plot', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Sample Predictions (adjusted for different data sizes)
    ax = axes[1, 2]
    
    # For SARIMA (time series data)
    if best['Model'] == 'SARIMA':
        sample = min(50, len(best_actual))
        x_range = range(sample)
        ax.plot(x_range, best_actual[:sample], 'o-', label='Actual', 
                linewidth=2, markersize=4, color='black')
        ax.plot(x_range, best_pred[:sample], 'x--', label='SARIMA', 
                alpha=0.7, linewidth=1.5, markersize=3, color='#2ecc71')
    else:
        # For ML models (tabular data)
        sample = 50
        x_range = range(sample)
        ax.plot(x_range, y_test.values[:sample], 'o-', label='Actual', 
                linewidth=2, markersize=4, color='black')
        
        colors_models = ['#3498db', '#e74c3c', '#2ecc71']
        for i, r in enumerate(results):
            if r['Model'] != 'SARIMA':
                ax.plot(x_range, r['predictions'][:sample], 'x--', 
                        label=r['Model'], alpha=0.7, linewidth=1.5, 
                        markersize=3, color=colors_models[i])
    
    ax.set_xlabel('Sample Index', fontweight='bold')
    ax.set_ylabel('Demand (units)', fontweight='bold')
    ax.set_title('First 50 Test Samples', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: evaluation_results.png")
    plt.show()
    
    # Save comparison table
    df_comp.to_csv('comparison_results.csv', index=False)
    print("✅ Saved: comparison_results.csv")
    
    print("\n" + "="*70)
    print("🎉 EVALUATION COMPLETE!")
    print("="*70)
    print(f"\n🏆 Best Model: {best['Model']}")
    print(f"📊 R² Score: {best['R2']:.4f}")
    print(f"📉 Average Error: {best['MAE']:.0f} units")
    
    # Insight on approach
    if best['Model'] == 'SARIMA':
        print("\n💡 Insight: Time series approach menang! SARIMA lebih baik menangkap")
        print("   pola temporal dan seasonality dalam data demand.")
    else:
        print("\n💡 Insight: ML approach menang! Model dengan features tambahan")
        print("   (harga, promo, dll) lebih akurat dari pure time series.")
    
    print("\n📁 Output Files:")
    print("   • evaluation_results.png")
    print("   • comparison_results.csv")