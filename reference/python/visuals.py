import numpy as np
import plotly.graph_objects as go
from dataset import extract_features

# Color palette for up to 5 classes
CLASS_COLORS = [
    '#E69F00', # Orange
    '#0072B2', # Blue
    '#009E73', # Green
    '#CC79A7', # Pink/Red
    '#D55E00'  # Dark Orange/Red
]

CLASS_COLORS_LIGHT = [
    'rgba(230, 159, 0, 0.4)',
    'rgba(0, 114, 178, 0.4)',
    'rgba(0, 158, 115, 0.4)',
    'rgba(204, 121, 167, 0.4)',
    'rgba(213, 94, 0, 0.4)'
]

def draw_network_svg(model):
    """
    Generates a beautifully styled dynamic SVG representing the neural network architecture.
    """
    width = 750
    height = 360
    layers_sizes = model.layer_sizes
    num_layers = len(layers_sizes)
    dx = (width - 160) / (num_layers - 1) if num_layers > 1 else 0
    
    coords = []
    for l_idx, size in enumerate(layers_sizes):
        x = 80 + l_idx * dx
        dy = min(40, (height - 60) / size) if size > 1 else 0
        layer_coords = []
        for n_idx in range(size):
            y = height / 2 + (n_idx - (size - 1) / 2) * dy
            layer_coords.append((x, y))
        coords.append(layer_coords)
        
    svg_elements = []
    
    for l_idx in range(num_layers - 1):
        layer_weights = model.layers[l_idx].weights
        n_prev = layers_sizes[l_idx]
        
        for i in range(n_prev):
            for j in range(layers_sizes[l_idx + 1]):
                w_val = layer_weights[i, j]
                x1, y1 = coords[l_idx][i]
                x2, y2 = coords[l_idx + 1][j]
                
                color = "#0072B2" if w_val > 0 else "#E69F00"
                thickness = 0.5 + 6.0 * (min(abs(w_val), 5.0) / 5.0)
                opacity = 0.15 + 0.70 * (min(abs(w_val), 5.0) / 5.0)
                
                line_svg = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" ' \
                           f'stroke="{color}" stroke-width="{thickness:.2f}" stroke-opacity="{opacity:.2f}" ' \
                           f'style="transition: all 0.3s ease;">' \
                           f'<title>Weight: {w_val:.3f}</title>' \
                           f'</line>'
                svg_elements.append(line_svg)
                
        if model.skip_connections and l_idx > 0:
            input_dim = layers_sizes[0]
            for s in range(input_dim):
                for j in range(layers_sizes[l_idx + 1]):
                    w_val = layer_weights[n_prev + s, j]
                    x1, y1 = coords[0][s]
                    x2, y2 = coords[l_idx + 1][j]
                    
                    color = "#0072B2" if w_val > 0 else "#E69F00"
                    thickness = 0.3 + 3.0 * (min(abs(w_val), 5.0) / 5.0)
                    opacity = 0.08 + 0.40 * (min(abs(w_val), 5.0) / 5.0)
                    
                    curve_height = 40.0 * l_idx
                    ctrl_x1 = x1 + (x2 - x1) * 0.25
                    ctrl_y1 = y1 - curve_height
                    ctrl_x2 = x2 - (x2 - x1) * 0.25
                    ctrl_y2 = y2 - curve_height
                    
                    path_svg = f'<path d="M {x1} {y1} C {ctrl_x1} {ctrl_y1}, {ctrl_x2} {ctrl_y2}, {x2} {y2}" ' \
                               f'fill="none" stroke="{color}" stroke-width="{thickness:.2f}" stroke-opacity="{opacity:.2f}" ' \
                               f'stroke-dasharray="3,3" style="transition: all 0.3s ease;">' \
                               f'<title>Skip Weight: {w_val:.3f}</title>' \
                               f'</path>'
                    svg_elements.append(path_svg)
                
    for l_idx, layer_coords in enumerate(coords):
        is_input = (l_idx == 0)
        is_output = (l_idx == num_layers - 1)
        
        if not is_input:
            biases = model.layers[l_idx - 1].biases[0]
        else:
            biases = None
            
        for n_idx, (x, y) in enumerate(layer_coords):
            r = 14
            
            if is_input:
                stroke_color = "#4A90E2"
                fill_color = "rgba(74, 144, 226, 0.15)"
                label_text = model.input_features[n_idx]
                tooltip = f"Input: {label_text}"
            elif is_output:
                bias_val = biases[n_idx]
                stroke_color = "#0072B2" if bias_val > 0 else "#E69F00"
                fill_color = "rgba(255, 255, 255, 0.1)"
                label_text = "Out" if model.problem_type == 'regression' or len(layer_coords) == 1 else f"C{n_idx}"
                tooltip = f"Output Neuron\nBias: {bias_val:.3f}"
            else:
                bias_val = biases[n_idx]
                stroke_color = "#0072B2" if bias_val > 0 else "#E69F00"
                fill_color = "rgba(255, 255, 255, 0.05)"
                label_text = f"h{l_idx}_{n_idx+1}"
                tooltip = f"Hidden Layer {l_idx}, Neuron {n_idx+1}\nBias: {bias_val:.3f}"
                
            circle_svg = f'<circle cx="{x}" cy="{y}" r="{r}" ' \
                         f'fill="{fill_color}" stroke="{stroke_color}" stroke-width="2.5" ' \
                         f'style="transition: all 0.3s ease; filter: drop-shadow(0px 0px 3px {stroke_color}aa);">' \
                         f'<title>{tooltip}</title>' \
                         f'</circle>'
            svg_elements.append(circle_svg)
            
            if is_input:
                text_svg = f'<text x="{x - 22}" y="{y + 4}" fill="#E0E0E0" font-size="10" font-family="sans-serif" font-weight="bold" text-anchor="end">{label_text}</text>'
                svg_elements.append(text_svg)
            elif is_output:
                text_svg = f'<text x="{x + 22}" y="{y + 4}" fill="#E0E0E0" font-size="10" font-family="sans-serif" font-weight="bold" text-anchor="start">{label_text}</text>'
                svg_elements.append(text_svg)
                
    for l_idx in range(num_layers):
        x = coords[l_idx][0][0]
        if l_idx == 0:
            header_text = "INPUTS"
        elif l_idx == num_layers - 1:
            header_text = "OUTPUTS"
        else:
            header_text = f"HIDDEN {l_idx}"
            
        header_svg = f'<text x="{x}" y="20" fill="#888888" font-size="11" font-family="sans-serif" font-weight="bold" text-anchor="middle">{header_text}</text>'
        svg_elements.append(header_svg)

    svg_container = f'<svg width="100%" height="100%" viewBox="0 0 {width} {height}" style="background-color: rgba(13, 17, 23, 0.4); border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.08); padding: 10px;">' \
                    f'{" ".join(svg_elements)}' \
                    f'</svg>'
    return svg_container

def plot_decision_boundary(model, X, y, dataset_type, problem_type, resolution=50):
    """
    Plots the decision boundary in Plotly.
    Fixed width and height are critical for flicker-free Streamlit rendering.
    """
    x_min, x_max = -6.0, 6.0
    y_min, y_max = -6.0, 6.0
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_features = extract_features(grid_points, model.input_features)
    preds = model.forward(grid_features)
    
    fig = go.Figure()
    
    if problem_type == 'classification':
        num_classes = model.layers[-1].output_dim
        if num_classes > 1:
            preds_class = np.argmax(preds, axis=1).reshape(xx.shape)
        else:
            preds_class = (preds > 0.5).astype(int).reshape(xx.shape)
            num_classes = 2

        colorscale = []
        for i in range(num_classes):
            norm_val = i / max(1, (num_classes - 1))
            colorscale.append([norm_val, CLASS_COLORS_LIGHT[i]])
            
        fig.add_trace(go.Contour(
            x=np.linspace(x_min, x_max, resolution),
            y=np.linspace(y_min, y_max, resolution),
            z=preds_class,
            colorscale=colorscale,
            showscale=False,
            zmin=0,
            zmax=max(1, num_classes - 1),
            line=dict(width=0),
            hoverinfo='skip'
        ))
        
        y_flat = np.argmax(y, axis=1) if len(y.shape) > 1 and y.shape[1] > 1 else y.ravel()
        for i in range(num_classes):
            mask = (y_flat == i)
            if np.any(mask):
                fig.add_trace(go.Scatter(
                    x=X[mask, 0],
                    y=X[mask, 1],
                    mode='markers',
                    marker=dict(
                        color=CLASS_COLORS[i],
                        size=8,
                        line=dict(width=1.0, color='#FFFFFF'),
                        opacity=0.9
                    ),
                    name=f'Class {i}',
                    showlegend=False
                ))
    else:
        preds = preds.reshape(xx.shape)
        colorscale = 'RdBu'
        zmin = float(np.min(preds))
        zmax = float(np.max(preds))
        
        fig.add_trace(go.Contour(
            x=np.linspace(x_min, x_max, resolution),
            y=np.linspace(y_min, y_max, resolution),
            z=preds,
            colorscale=colorscale,
            showscale=False,
            zmin=zmin,
            zmax=zmax,
            line=dict(width=0),
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode='markers',
            marker=dict(
                color=y.ravel(),
                colorscale=colorscale,
                size=8,
                line=dict(width=1.0, color='#FFFFFF'),
                showscale=False
            ),
            name='Data Points',
            showlegend=False
        ))

    fig.update_layout(
        xaxis=dict(
            range=[x_min, x_max], 
            gridcolor='rgba(255, 255, 255, 0.05)', 
            zerolinecolor='rgba(255, 255, 255, 0.1)',
            tickfont=dict(color='#888888'),
            fixedrange=True
        ),
        yaxis=dict(
            range=[y_min, y_max], 
            gridcolor='rgba(255, 255, 255, 0.05)', 
            zerolinecolor='rgba(255, 255, 255, 0.1)',
            tickfont=dict(color='#888888'),
            fixedrange=True
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        width=400,  # Fixed width to prevent auto-resize flickering
        height=390, # Fixed height to match exactly
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='closest',
        uirevision='constant'  # Prevents Plotly from flashing
    )
    
    return fig

def plot_combined_metrics(history, problem_key):
    """
    Plots the real-time training/testing Loss and Accuracy progression in a unified 340px tall figure.
    All 4 lines (Train Loss, Test Loss, Train Acc, Test Acc) are overlaid on the EXACT same axes.
    Y-axis is fixed from 0 to 1.1 to comfortably fit both Acc (0-1) and normalized Loss.
    X-axis is fixed to project forward, simulating real-time drawing.
    """
    fig = go.Figure()
    
    epochs = history.get('epochs', [])
    train_loss = history.get('train_loss', [])
    test_loss = history.get('test_loss', [])
    train_acc = history.get('train_acc', [])
    test_acc = history.get('test_acc', [])
    
    # Calculate a dynamic X-axis forward projection boundary (blocks of 500 epochs)
    current_max_epoch = epochs[-1] if len(epochs) > 0 else 0
    x_max = ((current_max_epoch // 500) + 1) * 500
    
    # Restore original colors requested by user:
    # Train Loss = Blue, Test Loss = Orange
    # Train Acc = Green, Test Acc = Pink
    loss_color_train = '#38bdf8'
    loss_color_test = '#fb923c'
    acc_color_train = '#34d399'
    acc_color_test = '#bc8cff'
    
    # ACCURACY LINES (Solid for Train, Dashed for Test)
    if problem_key == 'classification':
        fig.add_trace(go.Scatter(
            x=epochs, y=train_acc, mode='lines', name='Train Acc',
            line=dict(color=acc_color_train, width=2)
        ))
        fig.add_trace(go.Scatter(
            x=epochs, y=test_acc, mode='lines', name='Test Acc',
            line=dict(color=acc_color_test, width=2, dash='dash')
        ))
        
    # LOSS LINES
    fig.add_trace(go.Scatter(
        x=epochs, y=train_loss, mode='lines', name='Train Loss',
        line=dict(color=loss_color_train, width=2)
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=test_loss, mode='lines', name='Test Loss',
        line=dict(color=loss_color_test, width=2, dash='dash')
    ))
    
    fig.update_layout(
        xaxis=dict(
            title=dict(text='EPOCH', font=dict(color='#888888', size=10, family='sans-serif')), 
            gridcolor='rgba(255, 255, 255, 0.05)', 
            zerolinecolor='rgba(255, 255, 255, 0.1)',
            tickfont=dict(color='#888888'),
            range=[0, x_max] # Fixed X-axis projecting forward
        ),
        yaxis=dict(
            title=dict(text='METRICS', font=dict(color='#888888', size=10, family='sans-serif')), 
            gridcolor='rgba(255, 255, 255, 0.05)', 
            zerolinecolor='rgba(255, 255, 255, 0.1)',
            tickfont=dict(color='#888888'),
            range=[0, 1.1] # Fixed Y-axis from 0 to 1.1
        ),
        margin=dict(l=45, r=10, t=10, b=30),
        width=400,  # Fixed width to prevent auto-resize flickering
        height=390, # EXACT same height as the boundary plot
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation='h', 
            yanchor='bottom', 
            y=1.02, 
            xanchor='right', 
            x=1,
            font=dict(color='#E0E0E0', size=9),
            bgcolor='rgba(0,0,0,0)'
        ),
        uirevision='constant' # Crucial for 0-flicker updating
    )
    
    return fig
