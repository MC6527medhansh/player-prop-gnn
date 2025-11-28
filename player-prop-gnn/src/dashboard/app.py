"""
Streamlit Dashboard for Player Prop Predictions
Phase 4.4 - Interactive UI

Design Philosophy:
- Simple, demo-friendly interface
- Clear visual hierarchy
- Graceful error handling
- Fast (<1 second load time)
"""
import streamlit as st
import os
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from dashboard.utils import (
    check_health,
    get_metrics,
    predict_player,
    parse_metric,
    format_probability,
    format_confidence_interval,
    get_position_emoji,
    get_status_color,
    APIConnectionError,
    APITimeoutError,
    APIValidationError,
    APIServerError,
    APIError,
    API_URL
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Player Prop Predictions",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    /* Metric cards */
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    
    /* Status indicators */
    .status-healthy {
        color: green;
        font-weight: bold;
    }
    
    .status-unhealthy {
        color: red;
        font-weight: bold;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_api_connection():
    """
    Check if API is reachable and healthy.
    
    Returns:
        (bool, str): (is_healthy, message)
    """
    try:
        health = check_health(timeout=5)
        
        if health.status == 'healthy':
            return True, "✅ API is healthy"
        else:
            errors = ', '.join(health.errors) if health.errors else 'Unknown error'
            return False, f"⚠️ API unhealthy: {errors}"
    
    except APIConnectionError as e:
        return False, f"❌ Cannot connect to API at {API_URL}. Is it running?"
    
    except APITimeoutError:
        return False, f"⏱️ API health check timed out. API may be starting up."
    
    except Exception as e:
        return False, f"❌ Unexpected error: {e}"


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main dashboard application."""
    
    # ========================================
    # HEADER
    # ========================================
    
    st.markdown("<div class='main-header'>", unsafe_allow_html=True)
    st.title("⚽ Player Prop Predictions")
    st.markdown("*Bayesian Multi-Task Model with Uncertainty Quantification*")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # ========================================
    # API HEALTH CHECK
    # ========================================
    
    # Check API on startup
    is_healthy, health_message = check_api_connection()
    
    # Show API status in header
    if is_healthy:
        st.success(health_message)
    else:
        st.error(health_message)
        st.info(f"**Troubleshooting:**\n"
                f"1. Verify API is running: `docker-compose ps`\n"
                f"2. Check API logs: `docker-compose logs api`\n"
                f"3. Test API: `curl {API_URL}/health`")
        
        if st.button("🔄 Retry Connection"):
            st.rerun()
        
        st.stop()  # Don't render rest of app if API is down
    
    # ========================================
    # SIDEBAR: INPUT PANEL
    # ========================================
    
    with st.sidebar:
        st.header("🎮 Prediction Inputs")
        
        # Player information
        st.subheader("Player")
        player_id = st.number_input(
            "Player ID",
            min_value=1,
            value=5,
            step=1,
            help="Database player ID (positive integer)"
        )
        
        position = st.selectbox(
            "Position",
            ["Forward", "Midfielder", "Defender", "Goalkeeper"],
            help="Player's position on the field"
        )
        
        # Match context
        st.subheader("Match Context")
        opponent_id = st.number_input(
            "Opponent Team ID",
            min_value=1,
            value=42,
            step=1,
            help="Database opponent team ID"
        )
        
        was_home = st.checkbox(
            "Home Game",
            value=True,
            help="Is this player's team playing at home?"
        )
        
        # Features
        st.subheader("Recent Form")
        
        goals_rolling = st.slider(
            "Goals (5-game avg)",
            min_value=0.0,
            max_value=2.0,
            value=0.4,
            step=0.1,
            help="Average goals scored in last 5 matches"
        )
        
        shots_rolling = st.slider(
            "Shots on Target (5-game avg)",
            min_value=0.0,
            max_value=5.0,
            value=1.2,
            step=0.1,
            help="Average shots on target in last 5 matches"
        )
        
        opponent_strength = st.slider(
            "Opponent Strength",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help="0 = weak opponent, 1 = strong opponent"
        )
        
        days_since = st.number_input(
            "Days Since Last Match",
            min_value=0,
            max_value=30,
            value=7,
            step=1,
            help="Rest days since last match"
        )
        
        # Predict button
        st.markdown("---")
        predict_button = st.button("🔮 Get Predictions", type="primary", use_container_width=True)
    
    # ========================================
    # MAIN AREA: PREDICTIONS
    # ========================================
    
    if predict_button:
        # Build feature dict
        features = {
            "goals_rolling_5": goals_rolling,
            "shots_on_target_rolling_5": shots_rolling,
            "opponent_strength": opponent_strength,
            "days_since_last_match": float(days_since),
            "was_home": 1.0 if was_home else 0.0
        }
        
        # Make prediction
        try:
            with st.spinner("🔄 Running inference..."):
                response = predict_player(
                    player_id=player_id,
                    opponent_id=opponent_id,
                    position=position,
                    was_home=was_home,
                    features=features
                )
            
            # ========================================
            # DISPLAY PREDICTIONS
            # ========================================
            
            st.success(f"✅ Predictions generated in {response.inference_time_ms}ms")
            
            # Metadata row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Player ID", player_id)
            
            with col2:
                st.metric("Position", f"{get_position_emoji(position)} {position}")
            
            with col3:
                st.metric("Cached", "✅ Yes" if response.cached else "❌ No")
            
            with col4:
                st.metric("Inference Time", f"{response.inference_time_ms}ms")
            
            st.markdown("---")
            
            # Predictions row
            col1, col2, col3 = st.columns(3)
            
            # Goals
            with col1:
                st.subheader("⚽ Goals")
                goals = response.predictions['goals']
                
                st.metric(
                    "Expected Value",
                    f"{goals.mean:.3f}",
                    help="Mean of posterior distribution"
                )
                
                st.caption(f"Std: {goals.std:.3f}")
                st.caption(f"95% CI: {format_confidence_interval(goals.ci_low, goals.ci_high)}")
                
                st.markdown("**Probability:**")
                st.progress(goals.probability['0'], text=f"0 goals: {format_probability(goals.probability['0'])}")
                st.progress(goals.probability['1'], text=f"1 goal: {format_probability(goals.probability['1'])}")
                st.progress(goals.probability['2+'], text=f"2+ goals: {format_probability(goals.probability['2+'])}")
            
            # Shots
            with col2:
                st.subheader("🎯 Shots on Target")
                shots = response.predictions['shots']
                
                st.metric(
                    "Expected Value",
                    f"{shots.mean:.3f}",
                    help="Mean of posterior distribution"
                )
                
                st.caption(f"Std: {shots.std:.3f}")
                st.caption(f"95% CI: {format_confidence_interval(shots.ci_low, shots.ci_high)}")
                
                st.markdown("**Probability:**")
                st.progress(shots.probability['0'], text=f"0 shots: {format_probability(shots.probability['0'])}")
                st.progress(shots.probability['1'], text=f"1 shot: {format_probability(shots.probability['1'])}")
                st.progress(shots.probability['2+'], text=f"2+ shots: {format_probability(shots.probability['2+'])}")
            
            # Cards
            with col3:
                st.subheader("🟨 Cards")
                cards = response.predictions['cards']
                
                st.metric(
                    "Expected Value",
                    f"{cards.mean:.3f}",
                    help="Mean of posterior distribution"
                )
                
                st.caption(f"Std: {cards.std:.3f}")
                st.caption(f"95% CI: {format_confidence_interval(cards.ci_low, cards.ci_high)}")
                
                st.markdown("**Probability:**")
                st.progress(cards.probability['0'], text=f"No cards: {format_probability(cards.probability['0'])}")
                st.progress(cards.probability['1'], text=f"1 card: {format_probability(cards.probability['1'])}")
                st.progress(cards.probability['2+'], text=f"2+ cards: {format_probability(cards.probability['2+'])}")
            
        except APIValidationError as e:
            st.error(f"❌ Invalid input: {e}")
        
        except APIServerError as e:
            st.error(f"❌ Server error: {e}")
        
        except APITimeoutError:
            st.error("⏱️ Request timed out. API may be overloaded.")
        
        except APIConnectionError:
            st.error(f"❌ Lost connection to API at {API_URL}")
        
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
    
    else:
        # Show placeholder
        st.info("👈 Configure inputs in the sidebar and click **Get Predictions** to start")
    
    # ========================================
    # FOOTER: SYSTEM MONITORING
    # ========================================
    
    st.markdown("---")
    st.subheader("📊 System Monitoring")
    
    try:
        # Get metrics
        metrics_text = get_metrics(timeout=3)
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Request count
        with col1:
            request_count = parse_metric(metrics_text, "prop_api_requests_total")
            if request_count is not None:
                st.metric("Total Requests", f"{int(request_count)}")
            else:
                st.metric("Total Requests", "N/A")
        
        # Cache hit rate
        with col2:
            cache_hit_rate = parse_metric(metrics_text, "prop_api_cache_hit_rate")
            if cache_hit_rate is not None:
                st.metric("Cache Hit Rate", format_probability(cache_hit_rate))
            else:
                st.metric("Cache Hit Rate", "N/A")
        
        # Model loaded
        with col3:
            model_loaded = parse_metric(metrics_text, "prop_api_model_loaded")
            if model_loaded == 1.0:
                st.metric("Model Status", "✅ Loaded")
            else:
                st.metric("Model Status", "❌ Not Loaded")
        
        # Redis connected
        with col4:
            redis_connected = parse_metric(metrics_text, "prop_api_redis_connected")
            if redis_connected == 1.0:
                st.metric("Redis Status", "✅ Connected")
            else:
                st.metric("Redis Status", "⚠️ Disconnected")
    
    except Exception as e:
        st.warning(f"Could not fetch metrics: {e}")
    
    # Footer info
    st.caption(f"API URL: `{API_URL}` | Model Version: v1.0")


# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()