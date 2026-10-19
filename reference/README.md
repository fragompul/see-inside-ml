# Reference implementation (temporary)

The original Python/NumPy MLP engine and Streamlit dashboard. It is kept **only** as a
numerical oracle: in Phase 1 we generate JSON golden fixtures from it, and the TypeScript
engine (`packages/engine`) must reproduce them. Once the fixtures are committed and every
module is ported, this folder will be deleted.

Not part of the app. To run it anyway: `pip install -r python/requirements.txt && streamlit run python/app.py`.
