"""
UFC Fight Outcome Probability Prediction
Streamlit App for CIS 508 Machine Learning Final Project
Author: Anthony Nazaruk
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ===================================================================================
# PAGE CONFIG
# ===================================================================================
st.set_page_config(
    page_title="UFC Fight Predictor",
    page_icon="🥊",
    layout="centered"
)

# ===================================================================================
# LOAD MODEL AND DATA
# ===================================================================================
@st.cache_resource
def load_model():
    """Load the trained model pipeline."""
    with open('ufc_fight_predictor.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_fighter_data():
    """Load fighter statistics from the dataset."""
    df = pd.read_csv('ufc_fights_2019_2025_with_stats.csv')
    
    # Get the most recent stats for each fighter (they appear as fighter_a or fighter_b)
    # Create a unified fighter stats dataframe
    
    # Columns for fighter stats (without the fighter_a_ or fighter_b_ prefix)
    stat_cols = ['height', 'weight', 'age', 'reach', 'stance', 'style', 'country',
                 'total_fights', 'total_wins', 'win_rate', 'win_streak', 'loss_streak',
                 'ko_wins', 'sub_wins', 'dec_wins', 'recent_win_rate',
                 'avg_SSL', 'avg_SSA', 'avg_SS_ACC', 'avg_TSL', 'avg_TSA', 'avg_TS_ACC',
                 'avg_TDL', 'avg_TDA', 'avg_TD_ACC', 'avg_KD', 'avg_SM', 'avg_RV']
    
    # Extract fighter_a stats
    a_cols = {f'fighter_a_{col}': col for col in stat_cols}
    a_cols['fighter_a_name'] = 'name'
    a_cols['event_date'] = 'event_date'
    df_a = df[list(a_cols.keys())].rename(columns=a_cols)
    
    # Extract fighter_b stats
    b_cols = {f'fighter_b_{col}': col for col in stat_cols}
    b_cols['fighter_b_name'] = 'name'
    b_cols['event_date'] = 'event_date'
    df_b = df[list(b_cols.keys())].rename(columns=b_cols)
    
    # Combine and get most recent stats per fighter
    df_all = pd.concat([df_a, df_b], ignore_index=True)
    df_all['event_date'] = pd.to_datetime(df_all['event_date'])
    df_all = df_all.sort_values('event_date', ascending=False)
    
    # Keep most recent record for each fighter
    fighter_stats = df_all.groupby('name').first().reset_index()
    fighter_stats = fighter_stats.drop(columns=['event_date'])
    
    return fighter_stats

# ===================================================================================
# PREDICTION FUNCTION
# ===================================================================================
def prepare_matchup_features(fighter_a_stats, fighter_b_stats):
    """Prepare feature vector for a matchup between two fighters."""
    
    # Create the feature dictionary matching the training data structure
    features = {}
    
    # Fighter A stats
    for col in fighter_a_stats.index:
        if col != 'name':
            features[f'fighter_a_{col}'] = fighter_a_stats[col]
    
    # Fighter B stats
    for col in fighter_b_stats.index:
        if col != 'name':
            features[f'fighter_b_{col}'] = fighter_b_stats[col]
    
    # Calculate differential features
    features['height_diff'] = fighter_a_stats['height'] - fighter_b_stats['height']
    features['reach_diff'] = fighter_a_stats['reach'] - fighter_b_stats['reach']
    features['age_diff'] = fighter_a_stats['age'] - fighter_b_stats['age']
    features['weight_diff'] = fighter_a_stats['weight'] - fighter_b_stats['weight']
    features['experience_diff'] = fighter_a_stats['total_fights'] - fighter_b_stats['total_fights']
    features['win_rate_diff'] = fighter_a_stats['win_rate'] - fighter_b_stats['win_rate']
    features['recent_form_diff'] = fighter_a_stats['recent_win_rate'] - fighter_b_stats['recent_win_rate']
    features['win_streak_diff'] = fighter_a_stats['win_streak'] - fighter_b_stats['win_streak']
    
    # KO and Sub rate diffs (handle division by zero)
    a_ko_rate = fighter_a_stats['ko_wins'] / max(fighter_a_stats['total_wins'], 1)
    b_ko_rate = fighter_b_stats['ko_wins'] / max(fighter_b_stats['total_wins'], 1)
    features['ko_rate_diff'] = a_ko_rate - b_ko_rate
    
    a_sub_rate = fighter_a_stats['sub_wins'] / max(fighter_a_stats['total_wins'], 1)
    b_sub_rate = fighter_b_stats['sub_wins'] / max(fighter_b_stats['total_wins'], 1)
    features['sub_rate_diff'] = a_sub_rate - b_sub_rate
    
    # Striking and grappling stat diffs
    features['avg_SSL_diff'] = fighter_a_stats['avg_SSL'] - fighter_b_stats['avg_SSL']
    features['avg_SSA_diff'] = fighter_a_stats['avg_SSA'] - fighter_b_stats['avg_SSA']
    features['avg_SS_ACC_diff'] = fighter_a_stats['avg_SS_ACC'] - fighter_b_stats['avg_SS_ACC']
    features['avg_TSL_diff'] = fighter_a_stats['avg_TSL'] - fighter_b_stats['avg_TSL']
    features['avg_TSA_diff'] = fighter_a_stats['avg_TSA'] - fighter_b_stats['avg_TSA']
    features['avg_TS_ACC_diff'] = fighter_a_stats['avg_TS_ACC'] - fighter_b_stats['avg_TS_ACC']
    features['avg_TDL_diff'] = fighter_a_stats['avg_TDL'] - fighter_b_stats['avg_TDL']
    features['avg_TDA_diff'] = fighter_a_stats['avg_TDA'] - fighter_b_stats['avg_TDA']
    features['avg_TD_ACC_diff'] = fighter_a_stats['avg_TD_ACC'] - fighter_b_stats['avg_TD_ACC']
    features['avg_KD_diff'] = fighter_a_stats['avg_KD'] - fighter_b_stats['avg_KD']
    features['avg_SM_diff'] = fighter_a_stats['avg_SM'] - fighter_b_stats['avg_SM']
    features['avg_RV_diff'] = fighter_a_stats['avg_RV'] - fighter_b_stats['avg_RV']
    
    # Convert to DataFrame
    df = pd.DataFrame([features])
    
    # Handle missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('Unknown')
        else:
            df[col] = df[col].fillna(0)
    
    # Replace inf values
    df = df.replace([np.inf, -np.inf], 0)
    
    return df

# ===================================================================================
# MAIN APP
# ===================================================================================
def main():
    # Title
    st.title("🥊 UFC Fight Outcome Predictor")
    st.markdown("**CIS 508 Machine Learning Final Project** | Anthony Nazaruk")
    st.markdown("---")
    
    # Load model and data
    try:
        model = load_model()
        fighter_stats = load_fighter_data()
        fighter_names = sorted(fighter_stats['name'].dropna().unique().tolist())
    except FileNotFoundError as e:
        st.error(f"⚠️ Required file not found: {e}")
        st.info("Make sure 'ufc_fight_predictor.pkl' and 'ufc_fights_2019_2025_with_stats.csv' are in the app directory.")
        return
    
    st.markdown("### Select Fighters")
    st.markdown("Choose two fighters to predict the outcome of their matchup.")
    
    # Fighter selection dropdowns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔴 Fighter A")
        fighter_a_name = st.selectbox(
            "Select Fighter A",
            options=fighter_names,
            index=fighter_names.index("jon jones") if "jon jones" in fighter_names else 0,
            key="fighter_a"
        )
    
    with col2:
        st.markdown("#### 🔵 Fighter B")
        fighter_b_name = st.selectbox(
            "Select Fighter B",
            options=fighter_names,
            index=fighter_names.index("alex pereira") if "alex pereira" in fighter_names else 1,
            key="fighter_b"
        )
    
    # Warning if same fighter selected
    if fighter_a_name == fighter_b_name:
        st.warning("⚠️ Please select two different fighters.")
        return
    
    st.markdown("---")
    
    # Predict button
    if st.button("🎯 Predict Winner", type="primary", use_container_width=True):
        
        # Get fighter stats
        fighter_a_stats = fighter_stats[fighter_stats['name'] == fighter_a_name].iloc[0]
        fighter_b_stats = fighter_stats[fighter_stats['name'] == fighter_b_name].iloc[0]
        
        # Prepare features and predict
        with st.spinner("Analyzing matchup..."):
            features = prepare_matchup_features(fighter_a_stats, fighter_b_stats)
            
            try:
                prob_a_wins = model.predict_proba(features)[0][1]
                prob_b_wins = 1 - prob_a_wins
            except Exception as e:
                st.error(f"Prediction error: {e}")
                return
        
        # Display results
        st.markdown("---")
        st.markdown("### 📊 Prediction Results")
        
        # Determine favorite
        if prob_a_wins > prob_b_wins:
            favorite = fighter_a_name.title()
            favorite_prob = prob_a_wins
            underdog = fighter_b_name.title()
            underdog_prob = prob_b_wins
        else:
            favorite = fighter_b_name.title()
            favorite_prob = prob_b_wins
            underdog = fighter_a_name.title()
            underdog_prob = prob_a_wins
        
        # Results columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### 🔴 {fighter_a_name.title()}")
            st.metric(label="Win Probability", value=f"{prob_a_wins:.1%}")
            st.progress(prob_a_wins)
        
        with col2:
            st.markdown(f"#### 🔵 {fighter_b_name.title()}")
            st.metric(label="Win Probability", value=f"{prob_b_wins:.1%}")
            st.progress(prob_b_wins)
        
        # Winner announcement
        st.markdown("---")
        if abs(prob_a_wins - prob_b_wins) < 0.1:
            st.success(f"🤝 **CLOSE MATCHUP!** This fight is too close to call. Slight edge to **{favorite}** ({favorite_prob:.1%})")
        else:
            st.success(f"🏆 **PREDICTED WINNER: {favorite}** with {favorite_prob:.1%} probability")
        
        # Fighter stats comparison (expandable)
        with st.expander("📈 View Fighter Stats Comparison"):
            stats_df = pd.DataFrame({
                'Stat': ['Height', 'Weight', 'Reach', 'Age', 'Total Fights', 'Win Rate', 
                         'Win Streak', 'KO Wins', 'Sub Wins', 'Recent Form'],
                fighter_a_name.title(): [
                    f"{fighter_a_stats['height']:.1f}\"" if pd.notna(fighter_a_stats['height']) else "N/A",
                    f"{fighter_a_stats['weight']:.0f} lbs" if pd.notna(fighter_a_stats['weight']) else "N/A",
                    f"{fighter_a_stats['reach']:.1f}\"" if pd.notna(fighter_a_stats['reach']) else "N/A",
                    f"{fighter_a_stats['age']:.0f}" if pd.notna(fighter_a_stats['age']) else "N/A",
                    f"{fighter_a_stats['total_fights']:.0f}" if pd.notna(fighter_a_stats['total_fights']) else "N/A",
                    f"{fighter_a_stats['win_rate']:.1%}" if pd.notna(fighter_a_stats['win_rate']) else "N/A",
                    f"{fighter_a_stats['win_streak']:.0f}" if pd.notna(fighter_a_stats['win_streak']) else "N/A",
                    f"{fighter_a_stats['ko_wins']:.0f}" if pd.notna(fighter_a_stats['ko_wins']) else "N/A",
                    f"{fighter_a_stats['sub_wins']:.0f}" if pd.notna(fighter_a_stats['sub_wins']) else "N/A",
                    f"{fighter_a_stats['recent_win_rate']:.1%}" if pd.notna(fighter_a_stats['recent_win_rate']) else "N/A",
                ],
                fighter_b_name.title(): [
                    f"{fighter_b_stats['height']:.1f}\"" if pd.notna(fighter_b_stats['height']) else "N/A",
                    f"{fighter_b_stats['weight']:.0f} lbs" if pd.notna(fighter_b_stats['weight']) else "N/A",
                    f"{fighter_b_stats['reach']:.1f}\"" if pd.notna(fighter_b_stats['reach']) else "N/A",
                    f"{fighter_b_stats['age']:.0f}" if pd.notna(fighter_b_stats['age']) else "N/A",
                    f"{fighter_b_stats['total_fights']:.0f}" if pd.notna(fighter_b_stats['total_fights']) else "N/A",
                    f"{fighter_b_stats['win_rate']:.1%}" if pd.notna(fighter_b_stats['win_rate']) else "N/A",
                    f"{fighter_b_stats['win_streak']:.0f}" if pd.notna(fighter_b_stats['win_streak']) else "N/A",
                    f"{fighter_b_stats['ko_wins']:.0f}" if pd.notna(fighter_b_stats['ko_wins']) else "N/A",
                    f"{fighter_b_stats['sub_wins']:.0f}" if pd.notna(fighter_b_stats['sub_wins']) else "N/A",
                    f"{fighter_b_stats['recent_win_rate']:.1%}" if pd.notna(fighter_b_stats['recent_win_rate']) else "N/A",
                ]
            })
            st.table(stats_df)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 12px;'>
        Built with XGBoost & Streamlit | Model ROC-AUC: 0.719 | Data: UFC Fights 2019-2025
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
