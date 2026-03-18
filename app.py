"""
ExperimentLab — Open-Source Experimentation Platform
Streamlit dashboard for end-to-end experiment lifecycle.
"""

import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="ExperimentLab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports from src ─────────────────────────────────────────
from src.data.simulator import simulate_ab_test, simulate_geo_experiment, simulate_feedback_data
from src.data.schema import MetricDefinition, MetricType
from src.design.power import run_power_analysis, compute_mde
from src.design.geo_allocator import allocate_markets
from src.analysis.frequentist import run_frequentist_ab, check_srm
from src.analysis.bayesian import run_bayesian_ab
from src.analysis.sequential import run_sequential_analysis, compute_boundaries
from src.analysis.diff_in_diff import run_diff_in_diff
from src.analysis.synthetic_control import run_synthetic_control
from src.nlp.feedback import analyze_sentiment
from src.privacy.aggregation import run_privacy_audit
from src.metrics.builder import validate_metric, validate_metric_suite
from src.data.loader import (
    load_and_validate, validate_upload,
    AB_TEST_REQUIREMENTS, GEO_EXPERIMENT_REQUIREMENTS,
    FEEDBACK_REQUIREMENTS, PRIVACY_REQUIREMENTS,
)

# ── Helper: data source selector ────────────────────────────
def data_source_selector(tab_key, requirement, sample_path=None):
    """Show 'Upload CSV' / 'Use sample data' / 'Simulate' radio."""
    options = ["Simulate new data"]
    if sample_path:
        options.insert(0, "Use sample dataset")
    options.append("Upload your own CSV")

    source = st.radio(
        "Data source", options, horizontal=True, key=f"source_{tab_key}"
    )

    if source == "Upload your own CSV":
        st.markdown(f"**Required columns:** `{', '.join(requirement.required_cols)}`")
        if requirement.optional_cols:
            st.markdown(f"**Optional:** `{', '.join(requirement.optional_cols)}`")
        with st.expander("Example CSV format"):
            st.code(requirement.example_format, language="csv")
        uploaded = st.file_uploader("Upload CSV", type=["csv"], key=f"upload_{tab_key}")
        if uploaded:
            df, errors, warnings = load_and_validate(uploaded, requirement)
            for e in errors:
                st.error(e)
            for w in warnings:
                st.warning(w)
            if df is not None:
                st.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
                return df, "uploaded"
        return None, "uploaded"
    elif source == "Use sample dataset" and sample_path:
        try:
            df = pd.read_csv(sample_path)
            st.info(f"Using sample dataset: {len(df):,} rows")
            return df, "sample"
        except Exception as e:
            st.error(f"Could not load sample: {e}")
            return None, "sample"
    else:
        return None, "simulate"


# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.title("🧪 ExperimentLab")
st.sidebar.markdown("Open-source experimentation platform")
st.sidebar.markdown("---")

tabs = [
    "🏠 Overview",
    "⚡ Power Calculator",
    "🌍 Geo Allocator",
    "📊 Frequentist A/B",
    "🎲 Bayesian A/B",
    "🔄 Sequential Testing",
    "📈 Causal Methods",
    "💬 Sentiment Analysis",
    "🔒 Privacy Audit",
    "📐 Metric Validation",
]

selected_tab = st.sidebar.radio("Navigate", tabs, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Built for data scientists who run experiments.\n\n"
    "[GitHub](https://github.com) · [Docs](https://github.com)"
)


# ══════════════════════════════════════════════════════════════
# TAB 0: Overview
# ══════════════════════════════════════════════════════════════
if selected_tab == "🏠 Overview":
    st.title("🧪 ExperimentLab")
    st.markdown(
        "**End-to-end experimentation platform** — from power analysis to causal inference."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Analysis Methods", "10+")
    col2.metric("Statistical Tests", "8")
    col3.metric("Unit Tests", "147")
    col4.metric("Modules", "15")

    st.markdown("---")

    st.subheader("What You Can Do")
    cols = st.columns(3)

    with cols[0]:
        st.markdown("#### Pre-Experiment")
        st.markdown(
            "- **Power Calculator** — Sample size & MDE\n"
            "- **Geo Allocator** — Balanced market assignment\n"
            "- **Metric Validation** — Check metric health"
        )

    with cols[1]:
        st.markdown("#### During Experiment")
        st.markdown(
            "- **Sequential Testing** — Valid early stopping\n"
            "- **SRM Detection** — Randomization health\n"
            "- **Live Monitoring** — Boundary tracking"
        )

    with cols[2]:
        st.markdown("#### Post-Experiment")
        st.markdown(
            "- **Frequentist A/B** — z-test, t-test, CUPED\n"
            "- **Bayesian A/B** — Posterior inference\n"
            "- **DiD & Synth Control** — Causal methods\n"
            "- **Sentiment Analysis** — NLP feedback shift\n"
            "- **Privacy Audit** — k-anonymity & DP"
        )

    st.markdown("---")
    st.subheader("Quick Start: Simulate & Analyze")

    if st.button("🚀 Run Demo A/B Test", type="primary"):
        with st.spinner("Simulating 10,000 users..."):
            sim = simulate_ab_test(n_users=10000, true_effect=0.03, seed=42)
            df = sim.data
            c = df[df.group == 0].converted.values
            t = df[df.group == 1].converted.values

            freq = run_frequentist_ab(c, t, metric_type="binary")
            bayes = run_bayesian_ab(c, t, metric_type="binary")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Frequentist Result**")
            st.metric("Relative Lift", f"{freq.relative_lift:+.2%}")
            st.metric("p-value", f"{freq.p_value:.4f}")
            sig_text = "✅ Significant" if freq.significant else "❌ Not Significant"
            st.markdown(sig_text)

        with col_b:
            st.markdown("**Bayesian Result**")
            st.metric("P(Treatment Better)", f"{bayes.prob_treatment_better:.1%}")
            st.metric("Expected Loss", f"{bayes.expected_loss_treatment:.5f}")
            st.markdown(f"**Recommendation:** {bayes.recommendation.upper()}")


# ══════════════════════════════════════════════════════════════
# TAB 1: Power Calculator
# ══════════════════════════════════════════════════════════════
elif selected_tab == "⚡ Power Calculator":
    st.title("⚡ Power Calculator")
    st.markdown("How many users do you need? How long will the experiment run?")

    col1, col2 = st.columns(2)

    with col1:
        metric_type = st.selectbox("Metric Type", ["binary", "continuous"])
        baseline = st.number_input(
            "Baseline Rate" if metric_type == "binary" else "Baseline Mean",
            value=0.10 if metric_type == "binary" else 50.0,
            format="%.4f" if metric_type == "binary" else "%.2f",
        )
        mde = st.number_input(
            "Minimum Detectable Effect (absolute)",
            value=0.02 if metric_type == "binary" else 2.0,
            format="%.4f" if metric_type == "binary" else "%.2f",
        )

    with col2:
        alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, 0.01)
        power = st.slider("Statistical Power", 0.70, 0.99, 0.80, 0.01)
        daily_traffic = st.number_input("Daily Traffic (users/day)", value=5000, step=500)
        baseline_std = None
        if metric_type == "continuous":
            baseline_std = st.number_input("Baseline Std Dev", value=10.0)

    if st.button("Calculate", type="primary"):
        result = run_power_analysis(
            baseline_rate=baseline, mde=mde, alpha=alpha, power=power,
            metric_type=metric_type, daily_traffic=daily_traffic,
            baseline_std=baseline_std,
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Sample per Arm", f"{result.sample_per_arm:,}")
        c2.metric("Total Sample", f"{result.total_sample:,}")
        c3.metric("Duration", f"{result.estimated_days} days ({result.estimated_weeks} weeks)")

        # Power curve
        st.subheader("Power Curve")
        effects = np.linspace(mde * 0.2, mde * 3, 50)
        sample_sizes = []
        for e in effects:
            try:
                r = run_power_analysis(
                    baseline_rate=baseline, mde=e, alpha=alpha, power=power,
                    metric_type=metric_type, daily_traffic=daily_traffic,
                    baseline_std=baseline_std,
                )
                sample_sizes.append(r.sample_per_arm)
            except Exception:
                sample_sizes.append(None)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=effects, y=sample_sizes, mode="lines",
            name="Sample Size per Arm",
            line=dict(color="#636EFA", width=3),
        ))
        fig.add_vline(x=mde, line_dash="dash", line_color="red",
                       annotation_text=f"MDE = {mde}")
        fig.update_layout(
            xaxis_title="Effect Size",
            yaxis_title="Sample Size per Arm",
            template="plotly_white",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # MDE at different sample sizes
        st.subheader("Sensitivity Table")
        ns = [1000, 2500, 5000, 10000, 25000, 50000]
        mdes = []
        for n in ns:
            m = compute_mde(n, baseline, baseline_std, alpha, power, metric_type)
            mdes.append(m)
        st.dataframe(
            pd.DataFrame({"Sample per Arm": ns, "MDE": mdes}),
            use_container_width=True, hide_index=True,
        )


# ══════════════════════════════════════════════════════════════
# TAB 2: Geo Allocator
# ══════════════════════════════════════════════════════════════
elif selected_tab == "🌍 Geo Allocator":
    st.title("🌍 Geo Market Allocator")
    st.markdown("Balance markets for geo experiments.")

    col1, col2 = st.columns(2)
    with col1:
        n_markets = st.slider("Number of Markets", 10, 100, 30)
        treatment_frac = st.slider("Treatment Fraction", 0.1, 0.5, 0.3, 0.05)
    with col2:
        method = st.selectbox("Allocation Method", ["stratified", "random"])
        seed = st.number_input("Random Seed", value=42)

    if st.button("Allocate Markets", type="primary"):
        with st.spinner("Simulating market data and allocating..."):
            sim = simulate_geo_experiment(n_markets=n_markets, n_days=60, seed=seed)
            df = sim.data

            market_df = df.groupby("market_id").agg(
                avg_metric=("metric_value", "mean"),
                std_metric=("metric_value", "std"),
                total_metric=("metric_value", "sum"),
            ).reset_index()

            alloc = allocate_markets(
                market_df, treatment_fraction=treatment_frac,
                method=method, seed=seed,
            )

        c1, c2, c3 = st.columns(3)
        c1.metric("Treatment Markets", alloc.n_treatment)
        c2.metric("Control Markets", alloc.n_control)
        balance_status = "✅ Balanced" if alloc.balance.balanced else "⚠️ Imbalanced"
        c3.metric("Balance", balance_status)

        # Balance details
        st.subheader("Covariate Balance")
        balance_data = []
        for cov, stats in alloc.balance.covariate_stats.items():
            balance_data.append({
                "Covariate": cov,
                "Treatment Mean": stats["treatment_mean"],
                "Control Mean": stats["control_mean"],
                "Std Difference": stats["std_diff"],
            })
        st.dataframe(pd.DataFrame(balance_data), use_container_width=True, hide_index=True)

        # Market assignments
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Treatment Markets**")
            st.write(", ".join(alloc.treatment_markets))
        with col_b:
            st.markdown("**Control Markets**")
            st.write(", ".join(alloc.control_markets))


# ══════════════════════════════════════════════════════════════
# TAB 3: Frequentist A/B
# ══════════════════════════════════════════════════════════════
elif selected_tab == "📊 Frequentist A/B":
    st.title("📊 Frequentist A/B Test")

    # ── Data source ──
    uploaded_df, source = data_source_selector(
        "freq", AB_TEST_REQUIREMENTS, sample_path="data/samples/ab_test_sample.csv"
    )
    st.markdown("---")

    metric = st.selectbox("Metric", ["Conversion (binary)", "Revenue (continuous)"])
    alpha_freq = st.slider("α (significance level)", 0.01, 0.10, 0.05, 0.01, key="alpha_freq")

    if source == "simulate":
        col1, col2 = st.columns(2)
        with col1:
            n_users = st.slider("Total Users", 1000, 50000, 10000, 1000)
            true_effect = st.slider("True Effect (for simulation)", 0.0, 0.10, 0.03, 0.005)
        with col2:
            seed_freq = st.number_input("Seed", value=42, key="seed_freq")

    run_disabled = (source == "uploaded" and uploaded_df is None)
    if st.button("Run Frequentist Test", type="primary", disabled=run_disabled):
        with st.spinner("Analyzing..."):
            if source == "simulate":
                sim = simulate_ab_test(n_users=n_users, true_effect=true_effect, seed=seed_freq)
                df = sim.data
            elif source == "sample":
                df = uploaded_df
            else:
                df = uploaded_df

            if metric == "Conversion (binary)":
                c_data = df[df.group == 0].converted.values.astype(float)
                t_data = df[df.group == 1].converted.values.astype(float)
                mtype = "binary"
            else:
                col_name = "revenue" if "revenue" in df.columns else "converted"
                c_data = df[df.group == 0][col_name].values.astype(float)
                t_data = df[df.group == 1][col_name].values.astype(float)
                mtype = "continuous"

            result = run_frequentist_ab(c_data, t_data, metric_type=mtype, alpha=alpha_freq)
            srm = check_srm(len(c_data), len(t_data))

        # SRM check
        if srm["passed"]:
            st.success(f"✅ SRM Check Passed (p={srm['p_value']:.4f})")
        else:
            st.error(f"⚠️ SRM Detected! (p={srm['p_value']:.4f}) — Results may be unreliable")

        # Results
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Control Mean", f"{result.control_mean:.4f}")
        c2.metric("Treatment Mean", f"{result.treatment_mean:.4f}")
        c3.metric("Relative Lift", f"{result.relative_lift:+.2%}")
        c4.metric("p-value", f"{result.p_value:.4f}")

        sig_color = "green" if result.significant else "red"
        sig_text = "SIGNIFICANT" if result.significant else "NOT SIGNIFICANT"
        st.markdown(f"**Result: :{sig_color}[{sig_text}]** (α = {alpha_freq})")

        # CI visualization
        st.subheader("Confidence Interval")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[result.absolute_lift], y=["Lift"],
            orientation="h",
            error_x=dict(
                type="data",
                symmetric=False,
                array=[result.ci_upper - result.absolute_lift],
                arrayminus=[result.absolute_lift - result.ci_lower],
            ),
            marker_color="#636EFA" if result.significant else "#EF553B",
        ))
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(template="plotly_white", height=200, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Segment breakdown
        st.subheader("Segment Breakdown")
        segments = df.segment.unique()
        seg_data = []
        for seg in segments:
            seg_df = df[df.segment == seg]
            seg_c = seg_df[seg_df.group == 0].converted.values
            seg_t = seg_df[seg_df.group == 1].converted.values
            seg_data.append({
                "Segment": seg,
                "Control Rate": f"{seg_c.mean():.4f}",
                "Treatment Rate": f"{seg_t.mean():.4f}",
                "Lift": f"{(seg_t.mean() - seg_c.mean()):.4f}",
                "n": len(seg_df),
            })
        st.dataframe(pd.DataFrame(seg_data), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 4: Bayesian A/B
# ══════════════════════════════════════════════════════════════
elif selected_tab == "🎲 Bayesian A/B":
    st.title("🎲 Bayesian A/B Test")

    # ── Data source ──
    uploaded_df_b, source_b = data_source_selector(
        "bayes", AB_TEST_REQUIREMENTS, sample_path="data/samples/ab_test_sample.csv"
    )
    st.markdown("---")

    rope = st.slider("ROPE Width (±)", 0.001, 0.05, 0.01, 0.001)

    if source_b == "simulate":
        col1, col2 = st.columns(2)
        with col1:
            n_users_b = st.slider("Total Users", 1000, 50000, 10000, 1000, key="n_bayes")
            true_effect_b = st.slider("True Effect", 0.0, 0.10, 0.03, 0.005, key="eff_bayes")
        with col2:
            seed_b = st.number_input("Seed", value=42, key="seed_bayes")

    run_disabled_b = (source_b == "uploaded" and uploaded_df_b is None)
    if st.button("Run Bayesian Test", type="primary", disabled=run_disabled_b):
        with st.spinner("Running Bayesian inference..."):
            if source_b == "simulate":
                sim = simulate_ab_test(n_users=n_users_b, true_effect=true_effect_b, seed=seed_b)
                df = sim.data
            else:
                df = uploaded_df_b
            c = df[df.group == 0].converted.values.astype(float)
            t = df[df.group == 1].converted.values.astype(float)
            result = run_bayesian_ab(c, t, metric_type="binary", rope_width=rope)

        c1, c2, c3 = st.columns(3)
        c1.metric("P(Treatment Better)", f"{result.prob_treatment_better:.1%}")
        c2.metric("Expected Loss (Treatment)", f"{result.expected_loss_treatment:.5f}")
        c3.metric("Recommendation", result.recommendation.upper())

        st.metric("Relative Lift", f"{result.lift:+.2%}")
        st.markdown(
            f"**95% HDI on Lift:** [{result.lift_ci_lower:+.2%}, {result.lift_ci_upper:+.2%}]"
        )

        # Posterior distributions
        st.subheader("Posterior Distributions")
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Conversion Rate Posteriors", "Lift Distribution"])

        fig.add_trace(go.Histogram(
            x=result.control_posterior_samples, name="Control",
            opacity=0.6, marker_color="#636EFA", nbinsx=80,
        ), row=1, col=1)
        fig.add_trace(go.Histogram(
            x=result.treatment_posterior_samples, name="Treatment",
            opacity=0.6, marker_color="#EF553B", nbinsx=80,
        ), row=1, col=1)

        fig.add_trace(go.Histogram(
            x=result.lift_samples, name="Lift",
            marker_color="#00CC96", nbinsx=80, showlegend=False,
        ), row=1, col=2)
        fig.add_vline(x=0, line_dash="dash", line_color="gray", row=1, col=2)
        fig.add_vrect(x0=-rope, x1=rope, fillcolor="yellow", opacity=0.2,
                       annotation_text="ROPE", row=1, col=2)

        fig.update_layout(template="plotly_white", height=400, barmode="overlay")
        st.plotly_chart(fig, use_container_width=True)

        # Decision matrix
        st.subheader("Decision Summary")
        decision_data = {
            "Metric": [
                "P(Treatment > Control)",
                "P(Lift > 0)",
                "P(in ROPE)",
                "Expected Loss (Treatment)",
                "Expected Loss (Control)",
            ],
            "Value": [
                f"{result.prob_treatment_better:.1%}",
                f"{result.prob_lift_gt_0:.1%}",
                f"{result.prob_in_rope:.1%}",
                f"{result.expected_loss_treatment:.5f}",
                f"{result.expected_loss_control:.5f}",
            ],
        }
        st.dataframe(pd.DataFrame(decision_data), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 5: Sequential Testing
# ══════════════════════════════════════════════════════════════
elif selected_tab == "🔄 Sequential Testing":
    st.title("🔄 Sequential Testing")
    st.markdown("Valid early stopping with controlled Type I error.")

    col1, col2 = st.columns(2)
    with col1:
        n_users_seq = st.slider("Total Users", 2000, 50000, 10000, 1000, key="n_seq")
        true_eff_seq = st.slider("True Effect", 0.0, 0.10, 0.03, 0.005, key="eff_seq")
        n_looks = st.slider("Number of Looks", 2, 10, 5)
    with col2:
        spending = st.selectbox("Spending Function", ["obrien_fleming", "pocock"])
        alpha_seq = st.slider("α", 0.01, 0.10, 0.05, 0.01, key="alpha_seq")
        seed_seq = st.number_input("Seed", value=42, key="seed_seq")

    if st.button("Run Sequential Analysis", type="primary"):
        with st.spinner("Running sequential analysis..."):
            sim = simulate_ab_test(n_users=n_users_seq, true_effect=true_eff_seq, seed=seed_seq)
            df = sim.data
            c = df[df.group == 0].converted.values
            t = df[df.group == 1].converted.values
            result = run_sequential_analysis(
                c, t, n_looks=n_looks, alpha=alpha_seq,
                spending_function=spending,
            )

        early = "✅ Yes" if result.stopped_early else "❌ No"
        c1, c2, c3 = st.columns(3)
        c1.metric("Final Decision", result.final_decision)
        c2.metric("Stopped Early", early)
        c3.metric("Looks Conducted", len(result.looks))

        # Boundary plot
        st.subheader("Sequential Boundaries & Test Statistics")
        fig = go.Figure()

        looks_x = [b.information_fraction for b in result.boundaries]
        upper_y = [b.z_upper for b in result.boundaries]
        lower_y = [b.z_lower for b in result.boundaries]

        fig.add_trace(go.Scatter(
            x=looks_x, y=upper_y, mode="lines+markers",
            name="Efficacy Boundary", line=dict(color="green", width=2, dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=looks_x, y=lower_y, mode="lines+markers",
            name="Futility Boundary", line=dict(color="red", width=2, dash="dash"),
        ))

        # Actual z-statistics
        look_fracs = [l.boundary.information_fraction for l in result.looks]
        look_zs = [l.z_statistic for l in result.looks]
        fig.add_trace(go.Scatter(
            x=look_fracs, y=look_zs, mode="lines+markers",
            name="Observed z-stat", line=dict(color="#636EFA", width=3),
            marker=dict(size=10),
        ))

        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.update_layout(
            xaxis_title="Information Fraction",
            yaxis_title="z-statistic",
            template="plotly_white",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Looks table
        st.subheader("Interim Analysis Results")
        look_data = []
        for look in result.looks:
            look_data.append({
                "Look": look.look_number,
                "Info %": f"{look.boundary.information_fraction:.0%}",
                "n_control": look.n_control,
                "n_treatment": look.n_treatment,
                "z-stat": f"{look.z_statistic:+.3f}",
                "p-value": f"{look.p_value:.4f}",
                "Boundary (upper)": f"{look.boundary.z_upper:.3f}",
                "Decision": look.decision,
            })
        st.dataframe(pd.DataFrame(look_data), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 6: Causal Methods (DiD + Synthetic Control)
# ══════════════════════════════════════════════════════════════
elif selected_tab == "📈 Causal Methods":
    st.title("📈 Causal Inference Methods")

    method = st.selectbox("Method", ["Diff-in-Differences", "Synthetic Control"])

    # ── Data source ──
    uploaded_df_c, source_c = data_source_selector(
        "causal", GEO_EXPERIMENT_REQUIREMENTS, sample_path="data/samples/geo_experiment_sample.csv"
    )
    st.markdown("---")

    if source_c == "simulate":
        col1, col2 = st.columns(2)
        with col1:
            n_markets_c = st.slider("Markets", 10, 50, 20, key="causal_markets")
            n_days_c = st.slider("Days", 30, 180, 90, key="causal_days")
        with col2:
            effect_c = st.slider("True Effect", 0.0, 0.30, 0.15, 0.01, key="causal_effect")
            seed_c = st.number_input("Seed", value=42, key="causal_seed")

    run_disabled_c = (source_c == "uploaded" and uploaded_df_c is None)
    if st.button("Run Analysis", type="primary", disabled=run_disabled_c):
        if source_c == "simulate":
            with st.spinner(f"Simulating and running {method}..."):
                sim = simulate_geo_experiment(
                    n_markets=n_markets_c, n_days=n_days_c,
                    true_effect=effect_c, seed=seed_c,
                )
                df = sim.data
        else:
            df = uploaded_df_c

        if method == "Diff-in-Differences":
            result = run_diff_in_diff(df)

            c1, c2, c3 = st.columns(3)
            c1.metric("ATT (Treatment Effect)", f"{result.att:+.4f}")
            c2.metric("p-value", f"{result.p_value:.4f}")
            sig_text = "✅ Significant" if result.significant else "❌ Not Significant"
            c3.metric("Result", sig_text)

            st.markdown(
                f"**95% CI:** [{result.ci_lower:+.4f}, {result.ci_upper:+.4f}]"
            )

            pt = result.parallel_trends
            pt_status = "✅ PASSED" if pt.passed else "⚠️ FAILED"
            st.info(f"**Parallel Trends Test:** {pt_status} (p={pt.p_value:.4f})")

            # Event study plot
            if result.event_study is not None and len(result.event_study) > 0:
                st.subheader("Event Study (Dynamic Treatment Effects)")
                es = result.event_study
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=es.relative_period, y=es.effect,
                    mode="lines+markers", name="Effect",
                    line=dict(color="#636EFA", width=2),
                ))
                fig.add_trace(go.Scatter(
                    x=es.relative_period, y=es.ci_upper,
                    mode="lines", name="95% CI",
                    line=dict(width=0), showlegend=False,
                ))
                fig.add_trace(go.Scatter(
                    x=es.relative_period, y=es.ci_lower,
                    mode="lines", fill="tonexty",
                    line=dict(width=0), name="95% CI",
                    fillcolor="rgba(99, 110, 250, 0.2)",
                ))
                fig.add_vline(x=0, line_dash="dash", line_color="red",
                               annotation_text="Treatment Start")
                fig.add_hline(y=0, line_dash="dot", line_color="gray")
                fig.update_layout(
                    xaxis_title="Relative Period (days from treatment)",
                    yaxis_title="Treatment Effect",
                    template="plotly_white", height=450,
                )
                st.plotly_chart(fig, use_container_width=True)

        else:  # Synthetic Control
            if source_c == "simulate":
                treatment_markets = sim.config["treatment_markets"]
            else:
                treatment_markets = df[df.is_treatment == 1].market_id.unique().tolist()
            treatment_start = df[df.post_treatment == 1].date.min()

            result = run_synthetic_control(
                df, treatment_unit=treatment_markets[0],
                treatment_start=str(treatment_start),
                run_placebo=True,
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Estimated Effect", f"{result.estimated_effect:+.4f}")
            c2.metric("Effect (%)", f"{result.percent_effect:+.1f}%")
            c3.metric("RMSPE Ratio", f"{result.rmspe_ratio:.2f}")

            if result.placebo_p_value is not None:
                st.info(f"**Placebo p-value:** {result.placebo_p_value:.3f} "
                        f"(rank {result.placebo_rank}/{len(result.placebo_effects) + 1})")

            # Actual vs Synthetic plot
            st.subheader("Actual vs Synthetic Control")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                subplot_titles=["Actual vs Synthetic", "Gap (Effect)"],
                                vertical_spacing=0.12)

            fig.add_trace(go.Scatter(
                x=result.actual.index, y=result.actual.values,
                name="Actual", line=dict(color="#636EFA", width=2),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=result.synthetic.index, y=result.synthetic.values,
                name="Synthetic", line=dict(color="#EF553B", width=2, dash="dash"),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=result.gap.index, y=result.gap.values,
                name="Gap", line=dict(color="#00CC96", width=2),
                fill="tozeroy", fillcolor="rgba(0, 204, 150, 0.2)",
            ), row=2, col=1)

            fig.add_vline(x=pd.to_datetime(result.treatment_start),
                           line_dash="dash", line_color="red")
            fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)
            fig.update_layout(template="plotly_white", height=600)
            st.plotly_chart(fig, use_container_width=True)

            # Top donors
            st.subheader("Top Donor Weights")
            donor_df = pd.DataFrame(result.top_donors[:10], columns=["Market", "Weight"])
            st.dataframe(donor_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 7: Sentiment Analysis
# ══════════════════════════════════════════════════════════════
elif selected_tab == "💬 Sentiment Analysis":
    st.title("💬 Sentiment Analysis")
    st.markdown("Did your intervention change how users feel?")

    # ── Data source ──
    uploaded_df_nlp, source_nlp = data_source_selector(
        "nlp", FEEDBACK_REQUIREMENTS, sample_path="data/samples/feedback_sample.csv"
    )
    st.markdown("---")

    if source_nlp == "simulate":
        col1, col2 = st.columns(2)
        with col1:
            n_reviews = st.slider("Number of Reviews", 100, 2000, 500)
            sentiment_shift = st.slider("True Sentiment Shift", 0.0, 0.5, 0.2, 0.05)
        with col2:
            seed_nlp = st.number_input("Seed", value=42, key="seed_nlp")

    run_disabled_nlp = (source_nlp == "uploaded" and uploaded_df_nlp is None)
    if st.button("Analyze Sentiment", type="primary", disabled=run_disabled_nlp):
        with st.spinner("Analyzing sentiment..."):
            if source_nlp == "simulate":
                feedback = simulate_feedback_data(
                    n_reviews=n_reviews, true_sentiment_shift=sentiment_shift,
                )
            else:
                feedback = uploaded_df_nlp
            result = analyze_sentiment(feedback)

        c1, c2, c3 = st.columns(3)
        c1.metric("Pre-treatment Sentiment", f"{result.pre_mean_sentiment:+.3f}")
        c2.metric("Post-treatment Sentiment", f"{result.post_mean_sentiment:+.3f}")
        c3.metric("Shift", f"{result.sentiment_lift:+.3f}")

        sig_text = "✅ Significant" if result.significant else "❌ Not Significant"
        st.metric("p-value", f"{result.p_value:.4f}")
        st.markdown(sig_text)

        # Sentiment distribution
        st.subheader("Sentiment Composition")
        fig = go.Figure()
        categories = ["Positive", "Neutral", "Negative"]
        pre_vals = [result.pre_positive_pct, 100 - result.pre_positive_pct - result.pre_negative_pct, result.pre_negative_pct]
        post_vals = [result.post_positive_pct, 100 - result.post_positive_pct - result.post_negative_pct, result.post_negative_pct]

        fig.add_trace(go.Bar(name="Pre", x=categories, y=pre_vals, marker_color="#636EFA"))
        fig.add_trace(go.Bar(name="Post", x=categories, y=post_vals, marker_color="#EF553B"))
        fig.update_layout(barmode="group", template="plotly_white", height=350,
                           yaxis_title="Percentage")
        st.plotly_chart(fig, use_container_width=True)

        # Word shifts
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Top Words Increasing**")
            if result.top_positive_shift_words:
                for word, shift in result.top_positive_shift_words[:7]:
                    st.markdown(f"- `{word}` (+{shift:.4f})")
        with col_b:
            st.markdown("**Top Words Decreasing**")
            if result.top_negative_shift_words:
                for word, shift in result.top_negative_shift_words[:7]:
                    st.markdown(f"- `{word}` ({shift:.4f})")


# ══════════════════════════════════════════════════════════════
# TAB 8: Privacy Audit
# ══════════════════════════════════════════════════════════════
elif selected_tab == "🔒 Privacy Audit":
    st.title("🔒 Privacy Audit")
    st.markdown("k-anonymity checks and differential privacy.")

    # ── Data source ──
    uploaded_df_p, source_p = data_source_selector(
        "priv", PRIVACY_REQUIREMENTS, sample_path="data/samples/ab_test_sample.csv"
    )
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        k_threshold = st.slider("k-anonymity threshold", 3, 50, 10)
        if source_p == "simulate":
            n_users_p = st.slider("Users", 500, 10000, 2000, key="n_priv")
    with col2:
        epsilon = st.slider("ε (privacy budget)", 0.1, 10.0, 1.0, 0.1)
        apply_dp = st.checkbox("Apply Differential Privacy", value=True)

    run_disabled_p = (source_p == "uploaded" and uploaded_df_p is None)
    if st.button("Run Privacy Audit", type="primary", disabled=run_disabled_p):
        with st.spinner("Running audit..."):
            if source_p == "simulate":
                sim = simulate_ab_test(n_users=n_users_p, seed=42)
                df = sim.data
            else:
                df = uploaded_df_p

            # Auto-detect metric and group columns
            metric_col = "converted" if "converted" in df.columns else df.select_dtypes(include="number").columns[0]
            group_cols = [c for c in ["group", "segment"] if c in df.columns]
            if not group_cols:
                group_cols = [df.columns[0]]

            result = run_privacy_audit(
                df, metric_col=metric_col,
                group_cols=group_cols,
                apply_dp=apply_dp, epsilon=epsilon, k=k_threshold,
            )

        k_status = "✅ PASSED" if result.k_anonymity_passed else "⚠️ FAILED"
        c1, c2, c3 = st.columns(3)
        c1.metric("k-anonymity", k_status)
        c2.metric("Min Group Size", result.min_group_size)
        c3.metric("Groups Below k", f"{result.groups_below_threshold}/{result.total_groups}")

        if result.dp_applied:
            st.subheader("Differential Privacy")
            dp1, dp2, dp3 = st.columns(3)
            dp1.metric("Raw Mean", f"{result.raw_mean:.4f}")
            dp2.metric("Noisy Mean", f"{result.noisy_mean:.4f}")
            dp3.metric("Utility Loss", f"{result.utility_loss:.2%}")

            st.info(
                f"With ε={epsilon}, noise magnitude is {result.noise_magnitude:.4f}. "
                f"Lower ε = more privacy but less accuracy."
            )

        # Privacy-utility tradeoff
        st.subheader("Privacy-Utility Tradeoff")
        epsilons = np.arange(0.1, 10.1, 0.5)
        losses = []
        for eps in epsilons:
            r = run_privacy_audit(
                df, metric_col="converted",
                group_cols=["group", "segment"],
                apply_dp=True, epsilon=eps, k=k_threshold,
            )
            losses.append(r.utility_loss)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=epsilons, y=losses, mode="lines+markers",
            line=dict(color="#636EFA", width=2),
        ))
        fig.add_vline(x=epsilon, line_dash="dash", line_color="red",
                       annotation_text=f"Current ε={epsilon}")
        fig.update_layout(
            xaxis_title="ε (Privacy Budget)",
            yaxis_title="Utility Loss",
            template="plotly_white", height=350,
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 9: Metric Validation
# ══════════════════════════════════════════════════════════════
elif selected_tab == "📐 Metric Validation":
    st.title("📐 Metric Validation")
    st.markdown("Validate metric definitions against your data.")

    n_users_m = st.slider("Simulated Users", 1000, 20000, 5000, key="n_metric")

    if st.button("Validate Metrics", type="primary"):
        with st.spinner("Validating..."):
            sim = simulate_ab_test(n_users=n_users_m, seed=42)
            df = sim.data

            primary = MetricDefinition(
                name="conversion_rate",
                metric_type=MetricType.BINARY,
                numerator_col="converted",
                denominator_col=None,
            )
            guardrails = [
                MetricDefinition(
                    name="avg_revenue",
                    metric_type=MetricType.CONTINUOUS,
                    numerator_col="revenue",
                    denominator_col=None,
                ),
            ]

            suite = validate_metric_suite(
                df, primary=primary, guardrails=guardrails,
                sample_size=n_users_m // 2,
            )

        all_status = "✅ All Valid" if suite.all_valid else "⚠️ Issues Found"
        st.metric("Suite Status", all_status)

        st.subheader("Primary Metric")
        p = suite.primary
        pc1, pc2, pc3, pc4 = st.columns(4)
        pc1.metric("Name", p.metric_name)
        pc2.metric("Mean", f"{p.observed_mean:.4f}" if p.observed_mean else "N/A")
        pc3.metric("Std", f"{p.observed_std:.4f}" if p.observed_std else "N/A")
        pc4.metric("MDE (sensitivity)", f"{p.sensitivity:.4f}" if p.sensitivity else "N/A")

        if p.errors:
            for e in p.errors:
                st.error(e)
        if p.warnings:
            for w in p.warnings:
                st.warning(w)

        st.subheader("Guardrail Metrics")
        for g in suite.guardrails:
            status = "✅" if g.valid else "❌"
            st.markdown(f"{status} **{g.metric_name}** — Mean: {g.observed_mean}, Std: {g.observed_std}")
            if g.errors:
                for e in g.errors:
                    st.error(e)

        st.subheader("Validation Summary")
        st.code(suite.summary_text)
