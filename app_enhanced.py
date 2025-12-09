"""
UFC Fight Outcome Probability Prediction
Streamlit App for CIS 508 Machine Learning Final Project
Author: Anthony Nazaruk

Features:
- Symmetrical predictions (averages both fighter orderings)
- Betting edge calculator with live odds from BestFightOdds.com
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re

# ===================================================================================
# PAGE CONFIG
# ===================================================================================
st.set_page_config(
    page_title="UFC Fight Predictor",
    page_icon="🥊",
    layout="wide"
)

# ===================================================================================
# LOAD DATA AND TRAIN MODEL ON STARTUP
# ===================================================================================
@st.cache_resource
def load_model_and_data():
    """Load data and train model on startup."""
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
    
    df = pd.read_csv('ufc_fights_2019_2025_with_stats.csv')
    
    TARGET = "fighter_a_won"
    cols_to_drop = ['fight_key', 'fight_id', 'event_id', 'event_date', 'event_name', 'promotion',
                    'fighter_a_id', 'fighter_a_name', 'fighter_b_id', 'fighter_b_name',
                    'title_fight', 'fight_end_round', 'fight_result_type', 'fight_duration']
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df_model = df.drop(columns=cols_to_drop).copy()
    
    cat_cols = [c for c in df_model.columns if c != TARGET and df_model[c].dtype == 'object']
    num_cols = [c for c in df_model.columns if c != TARGET and df_model[c].dtype != 'object']
    
    y = df_model[TARGET].copy()
    X = df_model.drop(columns=[TARGET]).copy()
    
    for col in cat_cols: X[col] = X[col].fillna('Unknown')
    for col in num_cols: X[col] = X[col].fillna(X[col].median())
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in num_cols:
        X[col] = X[col].fillna(X[col].median())
        X[col] = X[col].clip(lower=-1e9, upper=1e9)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    preprocess = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])
    
    xgb = XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=4,
        reg_alpha=0, reg_lambda=5, random_state=42,
        use_label_encoder=False, eval_metric='logloss', n_jobs=-1
    )
    
    model = Pipeline([("prep", preprocess), ("clf", xgb)])
    model.fit(X_train, y_train)
    
    # Prepare fighter stats lookup
    stat_cols = ['height', 'weight', 'age', 'reach', 'stance', 'style', 'country',
                 'total_fights', 'total_wins', 'win_rate', 'win_streak', 'loss_streak',
                 'ko_wins', 'sub_wins', 'dec_wins', 'recent_win_rate',
                 'avg_SSL', 'avg_SSA', 'avg_SS_ACC', 'avg_TSL', 'avg_TSA', 'avg_TS_ACC',
                 'avg_TDL', 'avg_TDA', 'avg_TD_ACC', 'avg_KD', 'avg_SM', 'avg_RV']
    
    a_cols = {f'fighter_a_{col}': col for col in stat_cols}
    a_cols['fighter_a_name'] = 'name'
    a_cols['event_date'] = 'event_date'
    df_a = df[list(a_cols.keys())].rename(columns=a_cols)
    
    b_cols = {f'fighter_b_{col}': col for col in stat_cols}
    b_cols['fighter_b_name'] = 'name'
    b_cols['event_date'] = 'event_date'
    df_b = df[list(b_cols.keys())].rename(columns=b_cols)
    
    df_all = pd.concat([df_a, df_b], ignore_index=True)
    df_all['event_date'] = pd.to_datetime(df_all['event_date'])
    df_all = df_all.sort_values('event_date', ascending=False)
    fighter_stats = df_all.groupby('name').first().reset_index()
    fighter_stats = fighter_stats.drop(columns=['event_date'])
    
    return model, fighter_stats

# ===================================================================================
# PREDICTION FUNCTIONS
# ===================================================================================
def prepare_matchup_features(fighter_a_stats, fighter_b_stats):
    """Prepare feature vector for a matchup between two fighters."""
    features = {}
    
    for col in fighter_a_stats.index:
        if col != 'name':
            features[f'fighter_a_{col}'] = fighter_a_stats[col]
    
    for col in fighter_b_stats.index:
        if col != 'name':
            features[f'fighter_b_{col}'] = fighter_b_stats[col]
    
    features['height_diff'] = fighter_a_stats['height'] - fighter_b_stats['height']
    features['reach_diff'] = fighter_a_stats['reach'] - fighter_b_stats['reach']
    features['age_diff'] = fighter_a_stats['age'] - fighter_b_stats['age']
    features['weight_diff'] = fighter_a_stats['weight'] - fighter_b_stats['weight']
    features['experience_diff'] = fighter_a_stats['total_fights'] - fighter_b_stats['total_fights']
    features['win_rate_diff'] = fighter_a_stats['win_rate'] - fighter_b_stats['win_rate']
    features['recent_form_diff'] = fighter_a_stats['recent_win_rate'] - fighter_b_stats['recent_win_rate']
    features['win_streak_diff'] = fighter_a_stats['win_streak'] - fighter_b_stats['win_streak']
    
    a_ko_rate = fighter_a_stats['ko_wins'] / max(fighter_a_stats['total_wins'], 1)
    b_ko_rate = fighter_b_stats['ko_wins'] / max(fighter_b_stats['total_wins'], 1)
    features['ko_rate_diff'] = a_ko_rate - b_ko_rate
    
    a_sub_rate = fighter_a_stats['sub_wins'] / max(fighter_a_stats['total_wins'], 1)
    b_sub_rate = fighter_b_stats['sub_wins'] / max(fighter_b_stats['total_wins'], 1)
    features['sub_rate_diff'] = a_sub_rate - b_sub_rate
    
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
    
    df = pd.DataFrame([features])
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('Unknown')
        else:
            df[col] = df[col].fillna(0)
    
    df = df.replace([np.inf, -np.inf], 0)
    return df


def get_symmetric_prediction(model, fighter_stats, fighter_a_name, fighter_b_name):
    """
    Get symmetrical prediction by averaging both orderings.
    This ensures consistent probabilities regardless of which fighter is A vs B.
    """
    fighter_a_stats = fighter_stats[fighter_stats['name'] == fighter_a_name].iloc[0]
    fighter_b_stats = fighter_stats[fighter_stats['name'] == fighter_b_name].iloc[0]
    
    # Prediction 1: A vs B (probability that A wins)
    features_ab = prepare_matchup_features(fighter_a_stats, fighter_b_stats)
    prob_a_wins_v1 = model.predict_proba(features_ab)[0][1]
    
    # Prediction 2: B vs A (probability that B wins, which = 1 - prob A wins)
    features_ba = prepare_matchup_features(fighter_b_stats, fighter_a_stats)
    prob_b_wins_v2 = model.predict_proba(features_ba)[0][1]
    prob_a_wins_v2 = 1 - prob_b_wins_v2
    
    # Average both predictions for symmetry
    prob_a_wins = (prob_a_wins_v1 + prob_a_wins_v2) / 2
    prob_b_wins = 1 - prob_a_wins
    
    return prob_a_wins, prob_b_wins


# ===================================================================================
# ODDS SCRAPING AND CONVERSION
# ===================================================================================
def american_to_probability(odds):
    """Convert American odds to implied probability."""
    if odds is None:
        return None
    try:
        odds = float(odds)
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)
    except:
        return None


def parse_american_odds(odds_str):
    """Parse American odds string like '+150' or '-200' to integer."""
    if not odds_str or odds_str == '' or odds_str == 'n/a':
        return None
    try:
        # Remove arrows and other characters
        cleaned = re.sub(r'[▲▼]', '', odds_str).strip()
        if cleaned.startswith('+'):
            return int(cleaned[1:])
        elif cleaned.startswith('-'):
            return -int(cleaned[1:])
        else:
            return int(cleaned)
    except:
        return None


@st.cache_data(ttl=3600)  # Cache for 1 hour
def scrape_upcoming_ufc_odds():
    """Scrape upcoming UFC fight odds from BestFightOdds.com"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get('https://www.bestfightodds.com/', headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        events = []
        current_event = None
        current_fights = []
        
        # Find all event headers and table rows
        for element in soup.find_all(['a', 'tr']):
            # Check for event header
            if element.name == 'a' and element.get('href', '').startswith('/events/'):
                # Save previous event
                if current_event and current_fights:
                    events.append({'event': current_event, 'fights': current_fights})
                
                # Only get UFC events
                event_text = element.get_text(strip=True)
                if 'UFC' in event_text:
                    current_event = event_text
                    current_fights = []
                else:
                    current_event = None
                    
            # Check for fighter row (within UFC event)
            elif element.name == 'tr' and current_event:
                cells = element.find_all('td')
                fighter_link = element.find('a', href=lambda x: x and '/fighters/' in x)
                
                if fighter_link:
                    fighter_name = fighter_link.get_text(strip=True)
                    
                    # Get odds from cells (typically in columns after fighter name)
                    odds_values = []
                    for cell in cells[1:]:  # Skip first cell (fighter name)
                        cell_text = cell.get_text(strip=True)
                        if cell_text and cell_text != 'n/a':
                            parsed = parse_american_odds(cell_text)
                            if parsed is not None:
                                odds_values.append(parsed)
                    
                    # Calculate average odds if we have any
                    avg_odds = None
                    if odds_values:
                        avg_odds = sum(odds_values) / len(odds_values)
                    
                    current_fights.append({
                        'fighter': fighter_name,
                        'odds': avg_odds,
                        'implied_prob': american_to_probability(avg_odds)
                    })
        
        # Save last event
        if current_event and current_fights:
            events.append({'event': current_event, 'fights': current_fights})
        
        # Pair up fighters into matchups
        processed_events = []
        for event_data in events:
            fights = event_data['fights']
            matchups = []
            
            # Pair consecutive fighters (they're listed as opponents)
            for i in range(0, len(fights) - 1, 2):
                fighter_a = fights[i]
                fighter_b = fights[i + 1]
                
                # Only include if we have odds for at least one fighter
                if fighter_a['odds'] is not None or fighter_b['odds'] is not None:
                    matchups.append({
                        'fighter_a': fighter_a['fighter'],
                        'fighter_a_odds': fighter_a['odds'],
                        'fighter_a_implied': fighter_a['implied_prob'],
                        'fighter_b': fighter_b['fighter'],
                        'fighter_b_odds': fighter_b['odds'],
                        'fighter_b_implied': fighter_b['implied_prob']
                    })
            
            if matchups:
                processed_events.append({
                    'event': event_data['event'],
                    'matchups': matchups
                })
        
        return processed_events
    
    except Exception as e:
        st.error(f"Error fetching odds: {e}")
        return []


def normalize_name(name):
    """Normalize fighter name for matching."""
    if not name:
        return ""
    # Lowercase, remove extra spaces, handle common variations
    name = name.lower().strip()
    name = re.sub(r'\s+', ' ', name)
    # Remove common suffixes/prefixes
    name = re.sub(r'\s*(jr\.?|sr\.?|iii|ii)$', '', name)
    return name


def find_fighter_in_dataset(fighter_name, fighter_stats):
    """Try to find a fighter in our dataset using fuzzy matching."""
    normalized_search = normalize_name(fighter_name)
    
    for _, row in fighter_stats.iterrows():
        if normalize_name(row['name']) == normalized_search:
            return row['name']
    
    # Try partial matching (last name)
    search_parts = normalized_search.split()
    if search_parts:
        last_name = search_parts[-1]
        for _, row in fighter_stats.iterrows():
            row_parts = normalize_name(row['name']).split()
            if row_parts and row_parts[-1] == last_name:
                # Check if first name initial matches
                if len(search_parts) > 1 and len(row_parts) > 1:
                    if search_parts[0][0] == row_parts[0][0]:
                        return row['name']
    
    return None


# ===================================================================================
# MAIN APP
# ===================================================================================
def main():
    st.title("🥊 UFC Fight Outcome Predictor")
    st.markdown("**CIS 508 Machine Learning Final Project** | Anthony Nazaruk")
    
    # Load model and data
    with st.spinner("Loading model and fighter data..."):
        try:
            model, fighter_stats = load_model_and_data()
            fighter_names = sorted(fighter_stats['name'].dropna().unique().tolist())
        except Exception as e:
            st.error(f"⚠️ Error loading data: {e}")
            return
    
    # Create tabs for different features
    tab1, tab2 = st.tabs(["🎯 Custom Matchup Predictor", "💰 Upcoming Fights & Betting Edge"])
    
    # ===================================================================================
    # TAB 1: Custom Matchup Predictor
    # ===================================================================================
    with tab1:
        st.markdown("### Select Fighters")
        st.markdown("Choose two fighters to predict the outcome. Predictions are **symmetrical** — swapping fighters gives consistent probabilities.")
        
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
        
        if fighter_a_name == fighter_b_name:
            st.warning("⚠️ Please select two different fighters.")
            return
        
        st.markdown("---")
        
        if st.button("🎯 Predict Winner", type="primary", use_container_width=True):
            with st.spinner("Analyzing matchup..."):
                try:
                    prob_a_wins, prob_b_wins = get_symmetric_prediction(
                        model, fighter_stats, fighter_a_name, fighter_b_name
                    )
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    return
            
            st.markdown("### 📊 Prediction Results")
            
            if prob_a_wins > prob_b_wins:
                favorite = fighter_a_name.title()
                favorite_prob = prob_a_wins
            else:
                favorite = fighter_b_name.title()
                favorite_prob = prob_b_wins
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"#### 🔴 {fighter_a_name.title()}")
                st.metric(label="Win Probability", value=f"{prob_a_wins:.1%}")
                st.progress(float(prob_a_wins))
            
            with col2:
                st.markdown(f"#### 🔵 {fighter_b_name.title()}")
                st.metric(label="Win Probability", value=f"{prob_b_wins:.1%}")
                st.progress(float(prob_b_wins))
            
            st.markdown("---")
            if abs(prob_a_wins - prob_b_wins) < 0.1:
                st.success(f"🤝 **CLOSE MATCHUP!** Slight edge to **{favorite}** ({favorite_prob:.1%})")
            else:
                st.success(f"🏆 **PREDICTED WINNER: {favorite}** with {favorite_prob:.1%} probability")
            
            # Fighter stats comparison
            with st.expander("📈 View Fighter Stats Comparison"):
                fighter_a_stats_row = fighter_stats[fighter_stats['name'] == fighter_a_name].iloc[0]
                fighter_b_stats_row = fighter_stats[fighter_stats['name'] == fighter_b_name].iloc[0]
                
                stats_df = pd.DataFrame({
                    'Stat': ['Height', 'Weight', 'Reach', 'Age', 'Total Fights', 'Win Rate', 
                             'Win Streak', 'KO Wins', 'Sub Wins', 'Recent Form'],
                    fighter_a_name.title(): [
                        f"{fighter_a_stats_row['height']:.1f}\"" if pd.notna(fighter_a_stats_row['height']) else "N/A",
                        f"{fighter_a_stats_row['weight']:.0f} lbs" if pd.notna(fighter_a_stats_row['weight']) else "N/A",
                        f"{fighter_a_stats_row['reach']:.1f}\"" if pd.notna(fighter_a_stats_row['reach']) else "N/A",
                        f"{fighter_a_stats_row['age']:.0f}" if pd.notna(fighter_a_stats_row['age']) else "N/A",
                        f"{fighter_a_stats_row['total_fights']:.0f}" if pd.notna(fighter_a_stats_row['total_fights']) else "N/A",
                        f"{fighter_a_stats_row['win_rate']:.1%}" if pd.notna(fighter_a_stats_row['win_rate']) else "N/A",
                        f"{fighter_a_stats_row['win_streak']:.0f}" if pd.notna(fighter_a_stats_row['win_streak']) else "N/A",
                        f"{fighter_a_stats_row['ko_wins']:.0f}" if pd.notna(fighter_a_stats_row['ko_wins']) else "N/A",
                        f"{fighter_a_stats_row['sub_wins']:.0f}" if pd.notna(fighter_a_stats_row['sub_wins']) else "N/A",
                        f"{fighter_a_stats_row['recent_win_rate']:.1%}" if pd.notna(fighter_a_stats_row['recent_win_rate']) else "N/A",
                    ],
                    fighter_b_name.title(): [
                        f"{fighter_b_stats_row['height']:.1f}\"" if pd.notna(fighter_b_stats_row['height']) else "N/A",
                        f"{fighter_b_stats_row['weight']:.0f} lbs" if pd.notna(fighter_b_stats_row['weight']) else "N/A",
                        f"{fighter_b_stats_row['reach']:.1f}\"" if pd.notna(fighter_b_stats_row['reach']) else "N/A",
                        f"{fighter_b_stats_row['age']:.0f}" if pd.notna(fighter_b_stats_row['age']) else "N/A",
                        f"{fighter_b_stats_row['total_fights']:.0f}" if pd.notna(fighter_b_stats_row['total_fights']) else "N/A",
                        f"{fighter_b_stats_row['win_rate']:.1%}" if pd.notna(fighter_b_stats_row['win_rate']) else "N/A",
                        f"{fighter_b_stats_row['win_streak']:.0f}" if pd.notna(fighter_b_stats_row['win_streak']) else "N/A",
                        f"{fighter_b_stats_row['ko_wins']:.0f}" if pd.notna(fighter_b_stats_row['ko_wins']) else "N/A",
                        f"{fighter_b_stats_row['sub_wins']:.0f}" if pd.notna(fighter_b_stats_row['sub_wins']) else "N/A",
                        f"{fighter_b_stats_row['recent_win_rate']:.1%}" if pd.notna(fighter_b_stats_row['recent_win_rate']) else "N/A",
                    ]
                })
                st.table(stats_df)
    
    # ===================================================================================
    # TAB 2: Upcoming Fights & Betting Edge Calculator
    # ===================================================================================
    with tab2:
        st.markdown("### 💰 Betting Edge Calculator")
        st.markdown("Compare our model's predictions against sportsbook odds to find value bets.")
        
        # Edge threshold slider
        edge_threshold = st.slider(
            "🎯 Minimum Edge Threshold (%)",
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
            help="Fights with edge above this threshold are highlighted in green"
        )
        
        st.markdown("---")
        
        # Fetch odds
        with st.spinner("Fetching latest odds from BestFightOdds.com..."):
            events = scrape_upcoming_ufc_odds()
        
        if not events:
            st.warning("⚠️ No upcoming UFC events with odds found. Try again later.")
            return
        
        # Event selector
        event_names = [e['event'] for e in events]
        selected_event = st.selectbox("Select UFC Event", options=event_names)
        
        # Get matchups for selected event
        event_data = next((e for e in events if e['event'] == selected_event), None)
        
        if event_data:
            st.markdown(f"### {selected_event}")
            
            # Build results table
            results = []
            
            for matchup in event_data['matchups']:
                # Try to find fighters in our dataset
                fighter_a_dataset = find_fighter_in_dataset(matchup['fighter_a'], fighter_stats)
                fighter_b_dataset = find_fighter_in_dataset(matchup['fighter_b'], fighter_stats)
                
                row = {
                    'Fighter A': matchup['fighter_a'],
                    'Fighter B': matchup['fighter_b'],
                    'A Odds': f"{matchup['fighter_a_odds']:+.0f}" if matchup['fighter_a_odds'] else "N/A",
                    'B Odds': f"{matchup['fighter_b_odds']:+.0f}" if matchup['fighter_b_odds'] else "N/A",
                    'A Implied %': f"{matchup['fighter_a_implied']:.1%}" if matchup['fighter_a_implied'] else "N/A",
                    'B Implied %': f"{matchup['fighter_b_implied']:.1%}" if matchup['fighter_b_implied'] else "N/A",
                }
                
                # Get model predictions if both fighters found
                if fighter_a_dataset and fighter_b_dataset:
                    try:
                        model_prob_a, model_prob_b = get_symmetric_prediction(
                            model, fighter_stats, fighter_a_dataset, fighter_b_dataset
                        )
                        row['A Model %'] = f"{model_prob_a:.1%}"
                        row['B Model %'] = f"{model_prob_b:.1%}"
                        
                        # Calculate edge
                        if matchup['fighter_a_implied']:
                            edge_a = (model_prob_a - matchup['fighter_a_implied']) * 100
                            row['A Edge %'] = edge_a
                        else:
                            row['A Edge %'] = None
                            
                        if matchup['fighter_b_implied']:
                            edge_b = (model_prob_b - matchup['fighter_b_implied']) * 100
                            row['B Edge %'] = edge_b
                        else:
                            row['B Edge %'] = None
                            
                    except Exception:
                        row['A Model %'] = "N/A"
                        row['B Model %'] = "N/A"
                        row['A Edge %'] = None
                        row['B Edge %'] = None
                else:
                    row['A Model %'] = "Not in DB"
                    row['B Model %'] = "Not in DB"
                    row['A Edge %'] = None
                    row['B Edge %'] = None
                
                results.append(row)
            
            # Display results
            for row in results:
                with st.container():
                    st.markdown("---")
                    col1, col2, col3 = st.columns([2, 1, 2])
                    
                    with col1:
                        st.markdown(f"**🔴 {row['Fighter A']}**")
                        st.write(f"Odds: {row['A Odds']}")
                        st.write(f"Sportsbook Implied: {row['A Implied %']}")
                        st.write(f"Model Prediction: {row['A Model %']}")
                        
                        if row['A Edge %'] is not None:
                            edge = row['A Edge %']
                            if edge >= edge_threshold:
                                st.success(f"✅ Edge: **+{edge:.1f}%** (VALUE BET)")
                            elif edge > 0:
                                st.info(f"📊 Edge: +{edge:.1f}%")
                            else:
                                st.error(f"❌ Edge: {edge:.1f}%")
                    
                    with col2:
                        st.markdown("<h2 style='text-align: center;'>VS</h2>", unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"**🔵 {row['Fighter B']}**")
                        st.write(f"Odds: {row['B Odds']}")
                        st.write(f"Sportsbook Implied: {row['B Implied %']}")
                        st.write(f"Model Prediction: {row['B Model %']}")
                        
                        if row['B Edge %'] is not None:
                            edge = row['B Edge %']
                            if edge >= edge_threshold:
                                st.success(f"✅ Edge: **+{edge:.1f}%** (VALUE BET)")
                            elif edge > 0:
                                st.info(f"📊 Edge: +{edge:.1f}%")
                            else:
                                st.error(f"❌ Edge: {edge:.1f}%")
            
            # Summary of value bets
            st.markdown("---")
            st.markdown("### 📋 Value Bets Summary")
            
            value_bets = []
            for row in results:
                if row['A Edge %'] is not None and row['A Edge %'] >= edge_threshold:
                    value_bets.append({
                        'Fighter': row['Fighter A'],
                        'Odds': row['A Odds'],
                        'Model %': row['A Model %'],
                        'Implied %': row['A Implied %'],
                        'Edge': f"+{row['A Edge %']:.1f}%"
                    })
                if row['B Edge %'] is not None and row['B Edge %'] >= edge_threshold:
                    value_bets.append({
                        'Fighter': row['Fighter B'],
                        'Odds': row['B Odds'],
                        'Model %': row['B Model %'],
                        'Implied %': row['B Implied %'],
                        'Edge': f"+{row['B Edge %']:.1f}%"
                    })
            
            if value_bets:
                value_df = pd.DataFrame(value_bets)
                st.dataframe(value_df, use_container_width=True, hide_index=True)
            else:
                st.info(f"No value bets found with edge ≥ {edge_threshold}%. Try lowering the threshold.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 12px;'>
        Built with XGBoost & Streamlit | Model ROC-AUC: 0.719 | Data: UFC Fights 2019-2025<br>
        Odds data from BestFightOdds.com | Not financial advice
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
