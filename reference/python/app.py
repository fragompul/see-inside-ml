import streamlit as st
import numpy as np
import time

from dataset import generate_classification_data, generate_regression_data, extract_features
from neural_network import NeuralNetwork
from visuals import draw_network_svg, plot_decision_boundary, plot_combined_metrics

st.set_page_config(
    page_title="Neural Network Playground",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Space+Grotesk:wght@500;700&display=swap');
    
    /* ----------------------------------------------------------------------
       ANTI-FLICKER / ANTI-GRAY-OUT OVERRIDES
       Forces Streamlit to stay 100% visible and responsive during the loop 
       ---------------------------------------------------------------------- */
    [data-testid="stAppViewContainer"] {
        opacity: 1 !important;
        filter: none !important;
        transition: none !important;
    }
    [data-testid="stAppViewBlockContainer"] {
        opacity: 1 !important;
        filter: none !important;
        transition: none !important;
    }
    [data-testid="stStatusWidget"] {
        display: none !important;
        visibility: hidden !important;
    }

    /* Core aesthetics */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    .main-title {
        background: linear-gradient(135deg, #58a6ff 0%, #bc8cff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.2rem !important;
    }
    
    .subtitle {
        color: #8b949e;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .glass-card {
        background: rgba(22, 27, 34, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .glass-card:hover {
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 8px 32px 0 rgba(88, 166, 255, 0.05);
    }
    
    /* Custom HTML Stats Container to restore custom colors */
    .stat-container {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-around;
        align-items: center;
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        min-height: 85px; /* Fixed height to prevent layout jumps */
    }
    
    .stat-box {
        text-align: center;
        flex: 1;
        min-width: 90px;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
        margin: 5px 0;
    }
    
    .stat-box:last-child {
        border-right: none;
    }
    
    .stat-val {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(88, 166, 255, 0.15);
    }
    
    .stat-lbl {
        font-size: 0.75rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 5px;
        font-weight: 600;
    }
    
    div.row-widget.stRadio > div {
        flex-direction: row;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
    }
    
    .success-banner {
        background: rgba(46, 160, 67, 0.15);
        border: 1px solid rgba(46, 160, 67, 0.4);
        color: #3fb950;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
        margin-bottom: 20px;
        font-size: 1.1rem;
        font-family: 'Space Grotesk', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE INIT -----------------
if 'training_active' not in st.session_state:
    st.session_state.training_active = False
if 'epoch' not in st.session_state:
    st.session_state.epoch = 0
if 'step_clicked' not in st.session_state:
    st.session_state.step_clicked = False
if 'reset_clicked' not in st.session_state:
    st.session_state.reset_clicked = False
if 'last_lr' not in st.session_state:
    st.session_state.last_lr = 0.03
if 'auto_stopped' not in st.session_state:
    st.session_state.auto_stopped = False

st.markdown("<h1 class='main-title'>🧠 Neural Network Playground</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Visualize how artificial neural networks learn continuously in real-time.</p>", unsafe_allow_html=True)

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.markdown("### ⚙️ Training Optimizations")

batch_size = st.sidebar.selectbox("Batch Size", [10, 20, 40, 80, 150, 250], index=1)
optimizer = st.sidebar.selectbox("Optimizer", ["Adam", "RMSprop", "Momentum", "SGD"], index=0)
optimizer_key = {"Adam": "adam", "RMSprop": "rmsprop", "Momentum": "momentum", "SGD": "sgd"}[optimizer]

lr = st.sidebar.selectbox("Initial Learning Rate", [0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0], index=4)
lr_decay = st.sidebar.selectbox("LR Schedule", ["Constant", "Exponential Decay", "Cosine Annealing"], index=0)

activation = st.sidebar.selectbox("Activation Function", ["ReLU", "LeakyReLU", "Tanh", "Sigmoid", "ELU", "Swish", "Linear"], index=2)
activation_key = activation.lower().replace("leakyrelu", "leaky_relu")

reg_type = st.sidebar.selectbox("Regularization", ["None", "L1", "L2"], index=0)
reg_key = {"None": "none", "L1": "l1", "L2": "l2"}[reg_type]
reg_rate = st.sidebar.selectbox("Regularization Rate", [0.0, 0.0001, 0.001, 0.01, 0.1], index=0)

# ----------------- TOP ROW: CONFIGURATION -----------------
top_col1, top_col2, top_col3 = st.columns(3)

with top_col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 🛠️ General Settings")
    
    problem_type = st.selectbox("Problem Type", ["Classification", "Regression"], index=0)
    problem_key = 'classification' if problem_type == "Classification" else 'regression'
    
    if problem_key == 'classification':
        dataset_type = st.selectbox("Dataset", ["Circle", "XOR", "Gaussian Clusters", "Spiral"], index=3)
        dataset_map = {"Circle": "circle", "XOR": "xor", "Gaussian Clusters": "gaussian", "Spiral": "spiral"}
        num_classes = st.radio("Classes", options=[2, 3, 4, 5], index=0, horizontal=True)
        loss_key = "cce"
    else:
        dataset_type = st.selectbox("Dataset", ["Inclined Plane", "Saddle (Sin/Cos)"], index=1)
        dataset_map = {"Inclined Plane": "plane", "Saddle (Sin/Cos)": "saddle"}
        num_classes = 1
        loss_function = st.selectbox("Loss Function", ["MSE (Mean Squared Error)", "MAE", "Huber Loss"], index=0)
        loss_key = {"MSE (Mean Squared Error)": "default", "MAE": "mae", "Huber Loss": "huber"}[loss_function]
        
    dataset_key = dataset_map[dataset_type]
    
    noise = st.slider("Noise", 0.0, 1.0, 0.15, 0.05)
    train_ratio = st.slider("Training Ratio", 50, 90, 70, 5, format="%d%%")
        
    st.markdown('</div>', unsafe_allow_html=True)

with top_col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 🎯 Input Features")
    
    feat_col1, feat_col2 = st.columns(2)
    with feat_col1:
        x1_check = st.checkbox("X₁ (X coord)", value=True)
        x1_sq_check = st.checkbox("X₁²", value=False)
        sin_x1_check = st.checkbox("sin(X₁)", value=False)
        cos_x1_check = st.checkbox("cos(X₁)", value=False)
        sgn_x1_check = st.checkbox("sgn(X₁)", value=False)
        x1_x2_check = st.checkbox("X₁ · X₂", value=False)
    with feat_col2:
        x2_check = st.checkbox("X₂ (Y coord)", value=True)
        x2_sq_check = st.checkbox("X₂²", value=False)
        sin_x2_check = st.checkbox("sin(X₂)", value=False)
        cos_x2_check = st.checkbox("cos(X₂)", value=False)
        sgn_x2_check = st.checkbox("sgn(X₂)", value=False)
        rbf_check = st.checkbox("Exp(-r²/4)", value=False)
    
    selected_features = []
    if x1_check: selected_features.append("X1")
    if x2_check: selected_features.append("X2")
    if x1_sq_check: selected_features.append("X1^2")
    if x2_sq_check: selected_features.append("X2^2")
    if x1_x2_check: selected_features.append("X1*X2")
    if sin_x1_check: selected_features.append("sin(X1)")
    if sin_x2_check: selected_features.append("sin(X2)")
    if cos_x1_check: selected_features.append("cos(X1)")
    if cos_x2_check: selected_features.append("cos(X2)")
    if sgn_x1_check: selected_features.append("sgn(X1)")
    if sgn_x2_check: selected_features.append("sgn(X2)")
    if rbf_check: selected_features.append("RBF_Gaussian")
    
    if len(selected_features) == 0:
        st.warning("⚠️ Select at least one feature. Using X₁ and X₂ by default.")
        selected_features = ["X1", "X2"]
        
    st.markdown('</div>', unsafe_allow_html=True)

with top_col3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 🧱 Hidden Layers")
    
    num_hidden = st.radio("Number of Layers", options=[1, 2, 3, 4, 5, 6], index=1, horizontal=True)
    
    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 10px 0;'>", unsafe_allow_html=True)
    hidden_neurons = []
    for l in range(num_hidden):
        n_count = st.radio(f"Neurons in L{l+1}", options=[1, 2, 3, 4, 5, 6, 7, 8], index=3 if l < 2 else 1, horizontal=True, key=f"rad_l_{l}")
        hidden_neurons.append(n_count)
        
    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 15px 0;'>", unsafe_allow_html=True)
    skip_connections = st.checkbox("Skip Connections (ResNet style)", value=False)
    st.markdown('</div>', unsafe_allow_html=True)

current_config = {
    'problem_type': problem_key,
    'dataset_type': dataset_key,
    'noise': noise,
    'train_ratio': train_ratio,
    'selected_features': tuple(selected_features),
    'layer_neurons': tuple(hidden_neurons),
    'activation': activation_key,
    'skip_connections': skip_connections,
    'loss_type': loss_key,
    'optimizer': optimizer_key,
    'num_classes': num_classes
}

def initialize_playground(config):
    if config['problem_type'] == 'classification':
        X, y_labels = generate_classification_data(config['dataset_type'], n_samples=300, noise=config['noise'], num_classes=config['num_classes'])
        y = np.zeros((y_labels.size, config['num_classes']))
        y[np.arange(y_labels.size), y_labels] = 1.0
    else:
        X, y = generate_regression_data(config['dataset_type'], n_samples=300, noise=config['noise'])
        y = y.reshape(-1, 1)
    
    np.random.seed(42)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(len(X) * (config['train_ratio'] / 100.0))
    
    st.session_state.X_train = X[indices[:split]]
    st.session_state.y_train = y[indices[:split]]
    st.session_state.X_test = X[indices[split:]]
    st.session_state.y_test = y[indices[split:]]
    
    input_dim = len(config['selected_features'])
    if config['problem_type'] == 'classification':
        output_dim = config['num_classes']
        output_act = 'softmax'
    else:
        output_dim = 1
        output_act = 'linear'
        
    layer_sizes = [input_dim] + list(config['layer_neurons']) + [output_dim]
    activations = [config['activation']] * len(config['layer_neurons']) + [output_act]
    
    st.session_state.model = NeuralNetwork(
        layer_sizes=layer_sizes,
        activations=activations,
        input_features=list(config['selected_features']),
        problem_type=config['problem_type'],
        skip_connections=config['skip_connections'],
        loss_type=config['loss_type']
    )
    
    st.session_state.epoch = 0
    st.session_state.auto_stopped = False
    st.session_state.history = {
        'epochs': [],
        'train_loss': [],
        'test_loss': [],
        'train_acc': [],
        'test_acc': []
    }
    st.session_state.last_config = config

config_changed = ('last_config' not in st.session_state or 
                  st.session_state.last_config != current_config)

if config_changed or st.session_state.reset_clicked:
    initialize_playground(current_config)
    st.session_state.reset_clicked = False
    st.session_state.auto_stopped = False

# ----------------- TRAINING LOOP ENGINE -----------------
if lr_decay == "Exponential Decay":
    lr_t = lr * (0.96 ** (st.session_state.epoch / 100))
elif lr_decay == "Cosine Annealing":
    T_max = 500
    lr_min = lr * 0.01
    lr_t = lr_min + 0.5 * (lr - lr_min) * (1.0 + np.cos(np.pi * (st.session_state.epoch % T_max) / T_max))
else:
    lr_t = lr

st.session_state.last_lr = lr_t

if st.session_state.training_active or st.session_state.step_clicked:
    X_train_feat = extract_features(st.session_state.X_train, selected_features)
    X_test_feat = extract_features(st.session_state.X_test, selected_features)
    
    l1_rate = reg_rate if reg_key == 'l1' else 0.0
    l2_rate = reg_rate if reg_key == 'l2' else 0.0
    
    epochs_to_run = 1 if st.session_state.step_clicked else 10
    
    for _ in range(epochs_to_run):
        m = st.session_state.X_train.shape[0]
        indices = np.arange(m)
        np.random.shuffle(indices)
        X_shuffled = X_train_feat[indices]
        y_shuffled = st.session_state.y_train[indices]
        
        for b in range(0, m, batch_size):
            X_batch = X_shuffled[b:b+batch_size]
            y_batch = y_shuffled[b:b+batch_size]
            
            out_batch = st.session_state.model.forward(X_batch)
            st.session_state.model.backward(X_batch, y_batch, out_batch)
            
            st.session_state.model.update_weights(
                lr_t, 
                optimizer=optimizer_key, 
                l1_rate=l1_rate, 
                l2_rate=l2_rate
            )
            
        train_out = st.session_state.model.forward(X_train_feat)
        test_out = st.session_state.model.forward(X_test_feat)
        
        train_loss = st.session_state.model.compute_loss(train_out, st.session_state.y_train, l1_rate, l2_rate)
        test_loss = st.session_state.model.compute_loss(test_out, st.session_state.y_test, l1_rate, l2_rate)
        train_metric = st.session_state.model.compute_accuracy(train_out, st.session_state.y_train)
        test_metric = st.session_state.model.compute_accuracy(test_out, st.session_state.y_test)
        
        st.session_state.epoch += 1
        st.session_state.history['epochs'].append(st.session_state.epoch)
        st.session_state.history['train_loss'].append(train_loss)
        st.session_state.history['test_loss'].append(test_loss)
        st.session_state.history['train_acc'].append(train_metric)
        st.session_state.history['test_acc'].append(test_metric)
        
        if problem_key == 'classification' and test_metric >= 0.95:
            st.session_state.training_active = False
            st.session_state.auto_stopped = True
            break
            
    st.session_state.step_clicked = False


# ----------------- BOTTOM ROW: GRAPH & PLOTS -----------------
bot_col1, bot_col2 = st.columns([2.5, 3.5])

with bot_col1:
    # Controls
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.5, 1.2, 2.0])
    with ctrl_col1:
        if st.session_state.training_active:
            if st.button("⏸ Pause", use_container_width=True, type="secondary"):
                st.session_state.training_active = False
                st.session_state.auto_stopped = False
                st.rerun()
        else:
            if st.button("▶ Train", use_container_width=True, type="primary"):
                st.session_state.training_active = True
                st.session_state.auto_stopped = False
                st.rerun()
    with ctrl_col2:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.reset_clicked = True
            st.session_state.training_active = False
            st.rerun()
    with ctrl_col3:
        st.markdown(f"<div style='font-family: monospace; font-size: 1.3rem; padding-top: 6px; text-align: right; color: #8b949e;'>Epoch: <span style='color: #E0E0E0; font-weight: bold;'>{st.session_state.epoch:06d}</span></div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.session_state.auto_stopped:
        st.markdown("<div class='success-banner'>✅ Target 95% Accuracy Achieved! Training Halted.</div>", unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card" style="padding: 10px;">', unsafe_allow_html=True)
    st.markdown(f"<h5 style='margin-bottom: 5px; color: #8b949e;'>Architecture ({activation} • {optimizer})</h5>", unsafe_allow_html=True)
    
    st.markdown(draw_network_svg(st.session_state.model), unsafe_allow_html=True)
    # The restored legend for the connection weights
    st.markdown("<p style='text-align: center; font-size: 0.85rem; color: #8b949e; margin-top: 5px; margin-bottom: 5px;'>🟦 Positive Weights &nbsp;|&nbsp; 🟧 Negative Weights<br><i>Hover over neurons and connections to view exact values.</i></p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with bot_col2:
    train_loss = st.session_state.history['train_loss'][-1] if len(st.session_state.history['train_loss']) > 0 else 0.0
    test_loss = st.session_state.history['test_loss'][-1] if len(st.session_state.history['test_loss']) > 0 else 0.0
    train_metric = st.session_state.history['train_acc'][-1] if len(st.session_state.history['train_acc']) > 0 else 0.0
    test_metric = st.session_state.history['test_acc'][-1] if len(st.session_state.history['test_acc']) > 0 else 0.0

    metric_label = "Accuracy" if problem_key == 'classification' else "MSE"
    trainable_params = sum(layer.weights.size + layer.biases.size for layer in st.session_state.model.layers)

    if problem_key == 'classification':
        test_metric_str = f"{test_metric * 100:.1f}%"
    else:
        test_metric_str = f"{test_metric:.4f}"

    # Restore the beautiful custom colored HTML boxes
    stats_html = f"""
    <div class="stat-container">
        <div class="stat-box">
            <div class="stat-val" style="color: #38bdf8;">{train_loss:.3f}</div>
            <div class="stat-lbl">Train Loss</div>
        </div>
        <div class="stat-box">
            <div class="stat-val" style="color: #fb923c;">{test_loss:.3f}</div>
            <div class="stat-lbl">Test Loss</div>
        </div>
        <div class="stat-box">
            <div class="stat-val" style="color: #34d399;">{test_metric_str}</div>
            <div class="stat-lbl">Test {metric_label}</div>
        </div>
        <div class="stat-box">
            <div class="stat-val" style="color: #bc8cff;">{lr_t:.4f}</div>
            <div class="stat-lbl">LR</div>
        </div>
        <div class="stat-box">
            <div class="stat-val" style="color: #e2e8f0;">{trainable_params}</div>
            <div class="stat-lbl">Parameters</div>
        </div>
    </div>
    """
    st.markdown(stats_html, unsafe_allow_html=True)

    plot_col1, plot_col2 = st.columns(2)
    with plot_col1:
        fig_boundary = plot_decision_boundary(
            st.session_state.model, 
            st.session_state.X_train, 
            st.session_state.y_train, 
            dataset_key, 
            problem_key,
            resolution=50
        )
        # MUST disable use_container_width to prevent Plotly resizing layout shifts
        st.plotly_chart(
            fig_boundary, 
            use_container_width=False, 
            config={'displayModeBar': False},
            key="boundary_plot"
        )
        
    with plot_col2:
        fig_metrics = plot_combined_metrics(st.session_state.history, problem_key)
        # MUST disable use_container_width to prevent Plotly resizing layout shifts
        st.plotly_chart(
            fig_metrics, 
            use_container_width=False, 
            config={'displayModeBar': False},
            key="metrics_plot"
        )

# Loop continuation
if st.session_state.training_active:
    time.sleep(0.01)
    st.rerun()
