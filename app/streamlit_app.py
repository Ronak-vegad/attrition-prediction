from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib


def _load_artifacts(project_root: Path):
    models_dir = project_root / "models"
    pipeline_path = models_dir / "attrition_pipeline.pkl"
    threshold_path = models_dir / "threshold.json"
    schema_path = models_dir / "schema.json"

    if not pipeline_path.exists():
        st.error(
            "Model not found. Run the notebook first to export "
            "`models/attrition_pipeline.pkl`, `models/threshold.json`, and `models/schema.json`."
        )
        st.stop()

    pipeline = joblib.load(pipeline_path)
    threshold = json.loads(threshold_path.read_text(encoding="utf-8")) if threshold_path.exists() else {"threshold": 0.5}
    schema = json.loads(schema_path.read_text(encoding="utf-8")) if schema_path.exists() else None

    return pipeline, float(threshold.get("threshold", 0.5)), threshold, schema


def _build_input_form(schema: dict) -> dict:
    st.subheader("Employee profile")
    defaults = schema.get("defaults", {})

    inputs: dict[str, object] = {}

    # A nicer layout: show numeric + categorical in separate expanders
    with st.expander("Numeric features", expanded=True):
        for col in schema.get("numeric", {}):
            meta = schema["numeric"][col]
            vmin, vmax = meta.get("min", 0.0), meta.get("max", 1.0)
            default = defaults.get(col, (vmin + vmax) / 2)

            # Use number_input because some numeric fields are not naturally “sliderable”
            inputs[col] = st.number_input(
                label=col,
                value=float(default),
                min_value=float(vmin),
                max_value=float(vmax),
                step=1.0 if float(vmax - vmin) >= 10 else 0.1,
            )

    with st.expander("Categorical features", expanded=True):
        for col in schema.get("categorical", {}):
            options = schema["categorical"][col]
            default = defaults.get(col, options[0] if options else "")
            try:
                default_index = options.index(default)
            except ValueError:
                default_index = 0

            inputs[col] = st.selectbox(
                label=col,
                options=options,
                index=default_index,
            )

    return inputs


def _local_explanation(pipeline, input_df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame | None:
    """
    Best-effort local explanation:
    - For tree models: SHAP values (preferred)
    - For linear models: coefficient * feature value
    Falls back to None if anything fails.
    """
    try:
        import shap  # noqa: WPS433

        pre = pipeline.named_steps.get("preprocess")
        model = pipeline.named_steps.get("model")
        if pre is None or model is None:
            return None

        x_proc = pre.transform(input_df)
        feature_names = getattr(pre, "get_feature_names_out", lambda: None)()
        if feature_names is None:
            return None

        # Linear model explanation (fast)
        if hasattr(model, "coef_"):
            coefs = model.coef_.ravel()
            contrib = (x_proc[0] * coefs).ravel()
            out = pd.DataFrame({"feature": feature_names, "contribution": contrib})
            out["abs_contribution"] = out["contribution"].abs()
            return out.sort_values("abs_contribution", ascending=False).head(top_k)

        # Tree model: SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_proc)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # positive class

        contrib = np.array(shap_values[0]).ravel()
        out = pd.DataFrame({"feature": feature_names, "contribution": contrib})
        out["abs_contribution"] = out["contribution"].abs()
        return out.sort_values("abs_contribution", ascending=False).head(top_k)
    except Exception:
        return None


def main():
    project_root = Path(__file__).resolve().parents[1]
    pipeline, threshold, threshold_meta, schema = _load_artifacts(project_root)

    st.title("Employee Attrition Risk Predictor")
    st.caption("High-recall model + probability threshold tuned for flight-risk detection.")

    if schema is None:
        st.warning("Schema not found. Run the notebook to export `models/schema.json` for a guided form.")
        st.stop()

    inputs = _build_input_form(schema)

    st.divider()
    if st.button("Predict Attrition Risk", type="primary"):
        input_df = pd.DataFrame([inputs], columns=schema["columns"])
        proba = float(pipeline.predict_proba(input_df)[0, 1])
        pred = int(proba >= threshold)

        st.metric("Attrition Risk Score", f"{proba:.0%}")
        st.write(f"Decision threshold: **{threshold:.2f}** (tuned for recall target {threshold_meta.get('target_recall', 'N/A')})")

        if pred == 1:
            st.error("High risk — consider retention intervention.")
        else:
            st.success("Lower risk at the tuned threshold.")

        # Optional “driving factors” (best-effort local explanation)
        st.subheader("Top driving factors (best-effort)")
        explanation = _local_explanation(pipeline, input_df, top_k=10)
        if explanation is None or explanation.empty:
            st.info("Local explanation unavailable. See the notebook for SHAP-based global and local explanations.")
        else:
            st.dataframe(
                explanation[["feature", "contribution"]].reset_index(drop=True),
                use_container_width=True,
            )


if __name__ == "__main__":
    main()

