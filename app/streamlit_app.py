"""
Streamlit frontend for Steel Rebar Mechanical Property Prediction.

A polished dashboard with tabs for:
- Single Prediction (with quality gate badge)
- Batch Prediction (CSV upload)
- Model Insights (feature importance, comparison, residuals)
- Data Explorer (interactive distribution plots)
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    FEATURE_DESCRIPTIONS,
    FEATURE_RANGES,
    MODELS_DIR,
    OUTPUTS_DIR,
    VALID_DIAMETERS,
    VALID_GRADES,
    UTS_YS_MIN_RATIO,
    DATA_RAW_DIR,
    RAW_DATA_FILENAME,
)
from src.models.predictor import ModelRegistry, PredictionResult

# ─── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Steel Rebar Property Prediction",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


# ─── Initialize Registry ────────────────────────────────────
@st.cache_resource
def load_registry():
    return ModelRegistry()


registry = load_registry()


# ─── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔩 Configuration")
    st.markdown("---")

    diameter = st.selectbox(
        "**Rebar Diameter (mm)**",
        options=VALID_DIAMETERS,
        index=1,
        help="Select the rebar diameter to use the corresponding model.",
    )

    grade = st.selectbox(
        "**Steel Grade**",
        options=VALID_GRADES,
        index=0,
        help="Select the steel grade designation.",
    )

    st.markdown("---")

    # Model info
    available = registry.get_available_diameters()
    if available:
        st.markdown("### 📊 Available Models")
        for d in available:
            try:
                meta = registry.get_metadata(d)
                st.markdown(
                    f"- **{d}mm**: {meta['algorithm']} "
                    f"(R² = {meta['metrics']['combined_r2']:.3f})"
                )
            except Exception:
                st.markdown(f"- **{d}mm**: ⚠ Error loading")
    else:
        st.warning("No trained models found. Run `python train.py` first.")

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #6c757d; font-size: 0.8em;'>"
        "Built with Streamlit • scikit-learn • XGBoost"
        "</div>",
        unsafe_allow_html=True,
    )


# ─── Header ─────────────────────────────────────────────────
st.markdown("# 🏗️ Steel Rebar Property Prediction")
st.markdown(
    "Predict **Ultimate Tensile Strength (UTS)** and **Yield Strength (YS)** "
    "from manufacturing parameters in real time."
)

# ─── Tabs ────────────────────────────────────────────────────
tab_predict, tab_batch, tab_insights, tab_explorer = st.tabs(
    ["🎯 Predict", "📁 Batch Predict", "🧠 Model Insights", "📊 Data Explorer"]
)

# ═══════════════════════════════════════════════════════════════
# TAB 1: Single Prediction
# ═══════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown("### Enter Manufacturing Parameters")
    st.markdown(
        "Adjust the values below and click **Predict** to get "
        "predicted mechanical properties."
    )

    # Feature input columns
    col_chem, col_temp, col_proc = st.columns(3)

    features = {}

    with col_chem:
        st.markdown("#### 🧪 Chemical Composition")
        for col in [
            "CHEM1", "CHEM2", "CHEM3", "CHEM4", "CHEM5",
            "CHEM6", "CHEM7", "CHEM8", "CHEM9", "CHEM10",
        ]:
            mn, mx, default = FEATURE_RANGES[col]
            features[col] = st.number_input(
                f"{col} ({FEATURE_DESCRIPTIONS[col]})",
                min_value=float(mn),
                max_value=float(mx),
                value=float(default),
                format="%.4f" if mx < 1 else "%.2f" if mx < 200 else "%.1f",
                key=f"pred_{col}",
            )

    with col_temp:
        st.markdown("#### 🌡️ Temperatures")
        for col in ["TEMP1", "TEMP2", "TEMP3", "TEMP4", "TEMP5", "TEMP6"]:
            mn, mx, default = FEATURE_RANGES[col]
            features[col] = st.number_input(
                f"{col} ({FEATURE_DESCRIPTIONS[col]})",
                min_value=float(mn),
                max_value=float(mx),
                value=float(default),
                format="%.1f",
                key=f"pred_{col}",
            )

    with col_proc:
        st.markdown("#### ⚙️ Process Parameters")
        for col in ["SPEED", "PROCESS1", "PROCESS2", "PROCESS3"]:
            mn, mx, default = FEATURE_RANGES[col]
            features[col] = st.number_input(
                f"{col} ({FEATURE_DESCRIPTIONS[col]})",
                min_value=float(mn),
                max_value=float(mx),
                value=float(default),
                format="%.2f" if mx < 10 else "%.0f",
                key=f"pred_{col}",
            )

    st.markdown("---")

    # Predict button
    if st.button("🔮 **Predict Mechanical Properties**", use_container_width=True):
        try:
            result: PredictionResult = registry.predict(
                diameter=diameter,
                grade=grade,
                features=features.copy(),
            )

            # Results display
            st.markdown("### 📋 Prediction Results")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    label="UTS (QUALITY1)",
                    value=f"{result.quality1:.1f} MPa",
                )
            with col2:
                st.metric(
                    label="YS (QUALITY2)",
                    value=f"{result.quality2:.1f} MPa",
                )
            with col3:
                st.metric(
                    label="UTS/YS Ratio",
                    value=f"{result.uts_ys_ratio:.4f}",
                )
            with col4:
                if result.passes_quality_gate:
                    st.success(f"✅ **PASS**\nRatio ≥ {UTS_YS_MIN_RATIO}")
                else:
                    st.error(f"❌ **FAIL**\nRatio < {UTS_YS_MIN_RATIO}")

            # Gauge chart
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=result.uts_ys_ratio,
                    title={"text": "UTS/YS Ratio", "font": {"size": 20, "color": "white"}},
                    number={"font": {"size": 40, "color": "white"}},
                    delta={
                        "reference": UTS_YS_MIN_RATIO,
                        "increasing": {"color": "#00d26a"},
                        "decreasing": {"color": "#f44336"},
                    },
                    gauge={
                        "axis": {
                            "range": [0.9, 1.5],
                            "tickcolor": "white",
                            "tickfont": {"color": "white"},
                        },
                        "bar": {"color": "#3a7bd5"},
                        "bgcolor": "rgba(0,0,0,0)",
                        "steps": [
                            {"range": [0.9, UTS_YS_MIN_RATIO], "color": "rgba(244,67,54,0.3)"},
                            {"range": [UTS_YS_MIN_RATIO, 1.3], "color": "rgba(0,210,106,0.3)"},
                            {"range": [1.3, 1.5], "color": "rgba(0,210,255,0.3)"},
                        ],
                        "threshold": {
                            "line": {"color": "#ff6b6b", "width": 3},
                            "thickness": 0.8,
                            "value": UTS_YS_MIN_RATIO,
                        },
                    },
                )
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"},
                height=300,
                margin=dict(t=60, b=10, l=30, r=30),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.info(
                f"**Model used:** {result.model_name} | "
                f"**Diameter:** {result.diameter}mm | "
                f"**Grade:** {grade}"
            )

        except FileNotFoundError:
            st.error(
                f"⚠ No trained model found for diameter {diameter}mm. "
                f"Please run `python train.py` first."
            )
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")


# ═══════════════════════════════════════════════════════════════
# TAB 2: Batch Prediction
# ═══════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("### Upload CSV for Batch Predictions")
    st.markdown(
        "Upload a CSV file with the same columns as the input form above. "
        "The file must include `GRADE` and all feature columns. "
        "The `DIAMETER` column is optional — if absent, the sidebar selection is used."
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        key="batch_upload",
    )

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.markdown(f"**Loaded:** {len(batch_df)} rows × {len(batch_df.columns)} columns")
            st.dataframe(batch_df.head(10), use_container_width=True)

            if st.button("🚀 **Run Batch Predictions**", key="batch_predict"):
                results_list = []
                progress = st.progress(0)

                for idx, row in batch_df.iterrows():
                    row_dict = row.to_dict()
                    row_diameter = int(row_dict.pop("DIAMETER", diameter))
                    row_grade = row_dict.pop("GRADE", grade)

                    # Remove non-feature columns
                    for col in ["DATE_TIME", "ID", "QUALITY1", "QUALITY2",
                                "uts_ys_ratio", "uts/ys"]:
                        row_dict.pop(col, None)

                    try:
                        result = registry.predict(
                            diameter=row_diameter,
                            grade=row_grade,
                            features=row_dict,
                        )
                        results_list.append({
                            "Row": idx,
                            "Diameter": row_diameter,
                            "Grade": row_grade,
                            "Predicted_UTS": result.quality1,
                            "Predicted_YS": result.quality2,
                            "UTS_YS_Ratio": result.uts_ys_ratio,
                            "Quality_Gate": "PASS" if result.passes_quality_gate else "FAIL",
                        })
                    except Exception as e:
                        results_list.append({
                            "Row": idx,
                            "Diameter": row_diameter,
                            "Grade": row_grade,
                            "Predicted_UTS": None,
                            "Predicted_YS": None,
                            "UTS_YS_Ratio": None,
                            "Quality_Gate": f"ERROR: {str(e)[:50]}",
                        })

                    progress.progress((idx + 1) / len(batch_df))

                results_df = pd.DataFrame(results_list)
                st.markdown("### 📊 Batch Results")
                st.dataframe(results_df, use_container_width=True)

                # Summary stats
                valid = results_df[results_df["Predicted_UTS"].notna()]
                if len(valid) > 0:
                    pass_count = (valid["Quality_Gate"] == "PASS").sum()
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Samples", len(valid))
                    col2.metric("Pass Rate", f"{pass_count / len(valid) * 100:.1f}%")
                    col3.metric("Fail Count", len(valid) - pass_count)

                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")


# ═══════════════════════════════════════════════════════════════
# TAB 3: Model Insights
# ═══════════════════════════════════════════════════════════════
with tab_insights:
    st.markdown("### Model Performance & Insights")

    # Model comparison table
    st.markdown("#### 📊 Model Comparison Across Diameters")
    comparison_rows = []
    for d in VALID_DIAMETERS:
        try:
            meta = registry.get_metadata(d)
            m = meta["metrics"]
            comparison_rows.append({
                "Diameter": f"{d}mm",
                "Algorithm": meta["algorithm"],
                "R² (Combined)": f"{m['combined_r2']:.4f}",
                "R² (UTS)": f"{m.get('quality1_r2', 'N/A')}",
                "R² (YS)": f"{m.get('quality2_r2', 'N/A')}",
                "RMSE": f"{m['combined_rmse']:.2f}",
                "MAPE": f"{m['combined_mape']:.4f}",
                "Features": meta["feature_count"],
            })
        except Exception:
            pass

    if comparison_rows:
        st.dataframe(
            pd.DataFrame(comparison_rows),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")

    # Feature importance plots
    st.markdown("#### 🔍 Feature Importance")
    fi_col1, fi_col2, fi_col3 = st.columns(3)

    for col_widget, d in zip([fi_col1, fi_col2, fi_col3], VALID_DIAMETERS):
        fi_path = OUTPUTS_DIR / "feature_importance" / f"feature_importance_{d}mm.png"
        with col_widget:
            st.markdown(f"**Diameter {d}mm**")
            if fi_path.exists():
                st.image(str(fi_path), use_container_width=True)
            else:
                st.info(f"No plot found. Run training first.")

    st.markdown("---")

    # Actual vs Predicted
    st.markdown("#### 🎯 Actual vs Predicted")
    avp_col1, avp_col2, avp_col3 = st.columns(3)

    for col_widget, d in zip([avp_col1, avp_col2, avp_col3], VALID_DIAMETERS):
        avp_path = OUTPUTS_DIR / "model_comparison" / f"actual_vs_predicted_{d}mm.png"
        with col_widget:
            st.markdown(f"**Diameter {d}mm**")
            if avp_path.exists():
                st.image(str(avp_path), use_container_width=True)
            else:
                st.info("No plot found.")

    st.markdown("---")

    # Residual distributions
    st.markdown("#### 📈 Residual Distributions")
    res_col1, res_col2, res_col3 = st.columns(3)

    for col_widget, d in zip([res_col1, res_col2, res_col3], VALID_DIAMETERS):
        res_path = OUTPUTS_DIR / "residuals" / f"residuals_{d}mm.png"
        with col_widget:
            st.markdown(f"**Diameter {d}mm**")
            if res_path.exists():
                st.image(str(res_path), use_container_width=True)
            else:
                st.info("No plot found.")


# ═══════════════════════════════════════════════════════════════
# TAB 4: Data Explorer
# ═══════════════════════════════════════════════════════════════
with tab_explorer:
    st.markdown("### Explore Training Data Distributions")

    @st.cache_data
    def load_training_data():
        """Load the raw dataset for exploration."""
        path = DATA_RAW_DIR / RAW_DATA_FILENAME
        if path.exists():
            df = pd.read_csv(path)
            df = df.rename(columns={"PEOCESS3": "PROCESS3"})
            return df
        return None

    explore_df = load_training_data()

    if explore_df is not None:
        st.markdown(f"**Dataset:** {len(explore_df)} rows × {len(explore_df.columns)} columns")

        explore_col1, explore_col2 = st.columns(2)

        with explore_col1:
            # Distribution plot
            numeric_cols = explore_df.select_dtypes(include="number").columns.tolist()
            selected_col = st.selectbox(
                "Select feature to explore",
                options=numeric_cols,
                key="explore_feature",
            )

            fig = px.histogram(
                explore_df,
                x=selected_col,
                color="GRADE",
                marginal="box",
                title=f"Distribution of {selected_col} by Grade",
                template="plotly_dark",
                opacity=0.7,
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        with explore_col2:
            # Scatter plot
            x_col = st.selectbox("X-axis", options=numeric_cols, index=3, key="scatter_x")
            y_col = st.selectbox(
                "Y-axis",
                options=numeric_cols,
                index=numeric_cols.index("QUALITY1") if "QUALITY1" in numeric_cols else 0,
                key="scatter_y",
            )

            fig2 = px.scatter(
                explore_df,
                x=x_col,
                y=y_col,
                color="DIAMETER",
                title=f"{x_col} vs {y_col}",
                template="plotly_dark",
                opacity=0.6,
                color_continuous_scale="viridis",
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=400,
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Correlation heatmap
        st.markdown("#### 🔥 Correlation Heatmap")
        corr_cols = st.multiselect(
            "Select columns for correlation matrix",
            options=numeric_cols,
            default=[
                c for c in ["CHEM1", "CHEM2", "CHEM5", "TEMP1", "TEMP2",
                             "SPEED", "QUALITY1", "QUALITY2"]
                if c in numeric_cols
            ],
            key="corr_cols",
        )

        if len(corr_cols) >= 2:
            corr_matrix = explore_df[corr_cols].corr()
            fig3 = px.imshow(
                corr_matrix,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                title="Pearson Correlation Matrix",
                template="plotly_dark",
                zmin=-1,
                zmax=1,
            )
            fig3.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=500,
            )
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning(
            f"Training data not found at `{DATA_RAW_DIR / RAW_DATA_FILENAME}`. "
            f"Place the CSV file there to explore data."
        )
