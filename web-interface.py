import os
import yaml
import cv2
import numpy as np
import json
import uuid
import threading
import time
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename

# Import required modules from plotbot.py
from plotbot import (
    FilePlotter, SerialPlotter, CoordinateTransformer, 
    fit_image_on_canvas, draw_curve, compute_error, select_next_start
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
current_config = None
current_image = None
current_canvas = None
current_target_gray = None
rendering_active = False
plotting_active = False
session_id = None
progress = {
    'total_segments': 0,
    'current_error': 0,
    'status': 'idle',
    'percent_complete': 0
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_image, session_id
    
    # Generate a new session ID
    session_id = str(uuid.uuid4())
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Store the image path
        current_image = filepath
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath,
            'session_id': session_id
        })
    
    return jsonify({'error': 'File type not allowed'})

@app.route('/get_config', methods=['GET'])
def get_config():
    global current_config
    
    if current_config is None:
        current_config = load_config()
    
    return jsonify(current_config)

@app.route('/update_config', methods=['POST'])
def update_config():
    global current_config
    
    try:
        new_config = request.json
        
        # Save the new config to a file
        with open('config.yaml', 'w') as f:
            yaml.dump(new_config, f, default_flow_style=False)
        
        current_config = new_config
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/start_rendering', methods=['POST'])
def start_rendering():
    global rendering_active, current_image, current_config, current_canvas, current_target_gray, progress
    
    if rendering_active:
        return jsonify({'error': 'Rendering already in progress'})
    
    if current_image is None:
        return jsonify({'error': 'No image uploaded'})
    
    if current_config is None:
        current_config = load_config()
    
    # Reset progress
    progress = {
        'total_segments': 0,
        'current_error': 0,
        'status': 'initializing',
        'percent_complete': 0
    }
    
    # Start rendering in a background thread
    threading.Thread(target=render_image).start()
    
    return jsonify({'success': True})

@app.route('/start_plotting', methods=['POST'])
def start_plotting():
    global plotting_active, current_canvas, current_target_gray, current_config, progress
    
    if plotting_active:
        return jsonify({'error': 'Plotting already in progress'})
    
    if current_canvas is None or current_target_gray is None:
        return jsonify({'error': 'No rendering available. Please render an image first.'})
    
    # Start plotting in a background thread
    threading.Thread(target=plot_image).start()
    
    return jsonify({'success': True})

@app.route('/stop_rendering', methods=['POST'])
def stop_rendering():
    global rendering_active, progress
    
    rendering_active = False
    progress['status'] = 'stopped'
    
    return jsonify({'success': True})

@app.route('/stop_plotting', methods=['POST'])
def stop_plotting():
    global plotting_active, progress
    
    plotting_active = False
    progress['status'] = 'stopped'
    
    return jsonify({'success': True})

@app.route('/get_progress', methods=['GET'])
def get_progress():
    global progress
    return jsonify(progress)

@app.route('/get_preview', methods=['GET'])
def get_preview():
    global current_canvas, session_id
    
    if current_canvas is None:
        return jsonify({'error': 'No preview available'})
    
    # Save the current canvas as an image
    preview_filename = f"preview_{session_id}.jpg"
    preview_path = os.path.join(app.config['UPLOAD_FOLDER'], preview_filename)
    cv2.imwrite(preview_path, current_canvas)
    
    return jsonify({
        'success': True,
        'preview_url': f"/uploads/{preview_filename}?t={time.time()}"
    })

def render_image():
    global rendering_active, current_image, current_config, current_canvas, current_target_gray, progress
    
    rendering_active = True
    progress['status'] = 'rendering'
    
    try:
        # Load target image
        target_img = cv2.imread(current_image, cv2.IMREAD_GRAYSCALE)
        if target_img is None:
            progress['status'] = 'error'
            progress['error'] = 'Error loading image'
            rendering_active = False
            return
        
        # Canvas configuration
        canvas_cfg = current_config.get("canvas", {})
        canvas_width = canvas_cfg.get("width", 1000)
        canvas_height = canvas_cfg.get("height", 1000)
        
        # Fit image on canvas
        current_target_gray = fit_image_on_canvas(target_img, canvas_width, canvas_height)
        
        # Create white canvas (color)
        current_canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)
        
        # Scribble and segment parameters
        scribble_cfg = current_config.get('scribble', {})
        segment_cfg = current_config.get('segment', {})
        
        scribble_min = scribble_cfg.get('min_segments', 5)
        scribble_max = scribble_cfg.get('max_segments', 200)
        improvement_threshold = scribble_cfg.get('improvement_threshold', 0.01)
        seg_length_min = segment_cfg.get('length_min', 5)
        seg_length_max = segment_cfg.get('length_max', 30)
        delta_min = segment_cfg.get('delta_min', -np.pi/2)
        delta_max = segment_cfg.get('delta_max', np.pi/2)
        stroke_thickness = current_config.get('stroke', {}).get('thickness', 3)
        colors = current_config.get('colors', [[0, 0, 0]])
        seg_candidates = current_config.get('candidates', {}).get('segment', 20)
        start_candidates = current_config.get('candidates', {}).get('start', 100)
        
        # Initial error
        initial_error = compute_error(current_canvas, current_target_gray)
        current_error = initial_error
        
        total_segments = 0
        max_segments = 1000000  # Increased from 1000 to 1000000
        
        # Main rendering loop
        while rendering_active and total_segments < max_segments:
            # Choose new starting point
            start_x, start_y = select_next_start(current_canvas, current_target_gray, num_candidates=start_candidates)
            current_theta = np.random.uniform(0, 2*np.pi)
            current_x, current_y = start_x, start_y
            scribble_color = tuple(int(c) for c in colors[0])  # Use first color for now
            
            scribble_start_error = compute_error(current_canvas, current_target_gray)
            scribble_count = 0
            
            while rendering_active and scribble_count < scribble_max:
                best_error = None
                best_canvas = None
                best_endpoint = (None, None)
                best_new_theta = None
                
                for _ in range(seg_candidates):
                    L = np.random.uniform(seg_length_min, seg_length_max)
                    delta = np.random.uniform(delta_min, delta_max)
                    candidate_canvas, candidate_x, candidate_y, candidate_theta, _ = draw_curve(
                        current_canvas, current_x, current_y, current_theta, L, delta, scribble_color, stroke_thickness
                    )
                    err = compute_error(candidate_canvas, current_target_gray)
                    
                    if best_error is None or err < best_error:
                        best_error = err
                        best_canvas = candidate_canvas
                        best_endpoint = (candidate_x, candidate_y)
                        best_new_theta = candidate_theta
                
                # If no improvement found, skip this scribble
                if best_error is None or best_error >= current_error:
                    break
                
                current_canvas = best_canvas
                current_x, current_y = best_endpoint
                current_theta = best_new_theta
                current_error = best_error
                
                scribble_count += 1
                total_segments += 1
                
                # Update progress
                progress['total_segments'] = total_segments
                progress['current_error'] = int(current_error)
                progress['percent_complete'] = min(100, int(100 * (1 - current_error / initial_error)))
                
                # Save preview periodically
                if total_segments % 10 == 0:
                    preview_filename = f"preview_{session_id}.jpg"
                    preview_path = os.path.join(app.config['UPLOAD_FOLDER'], preview_filename)
                    cv2.imwrite(preview_path, current_canvas)
                
                # Check for stopping conditions
                new_err = compute_error(current_canvas, current_target_gray)
                improvement = scribble_start_error - new_err
                if scribble_count >= scribble_min and improvement < scribble_start_error * improvement_threshold:
                    break
        
        progress['status'] = 'completed'
        
    except Exception as e:
        progress['status'] = 'error'
        progress['error'] = str(e)
    
    rendering_active = False

def plot_image():
    global plotting_active, current_canvas, current_target_gray, current_config, progress
    
    plotting_active = True
    progress['status'] = 'plotting'
    
    try:
        # Canvas configuration
        canvas_cfg = current_config.get("canvas", {})
        canvas_width = canvas_cfg.get("width", 1000)
        canvas_height = canvas_cfg.get("height", 1000)
        
        # Plotter configuration
        plotter_cfg = current_config.get("plotter", {})
        
        # Initialize plotter
        try:
            plotter = SerialPlotter(plotter_cfg)
            plotter.initialize()
        except Exception as e:
            progress['status'] = 'error'
            progress['error'] = f"Error initializing plotter: {str(e)}"
            plotting_active = False
            return
        
        # Initialize coordinate transformer
        transformer = CoordinateTransformer(canvas_width, canvas_height, plotter_cfg)
        
        # Scribble and segment parameters
        scribble_cfg = current_config.get('scribble', {})
        segment_cfg = current_config.get('segment', {})
        
        scribble_min = scribble_cfg.get('min_segments', 5)
        scribble_max = scribble_cfg.get('max_segments', 200)
        improvement_threshold = scribble_cfg.get('improvement_threshold', 0.01)
        seg_length_min = segment_cfg.get('length_min', 5)
        seg_length_max = segment_cfg.get('length_max', 30)
        delta_min = segment_cfg.get('delta_min', -np.pi/2)
        delta_max = segment_cfg.get('delta_max', np.pi/2)
        stroke_thickness = current_config.get('stroke', {}).get('thickness', 3)
        colors = current_config.get('colors', [[0, 0, 0]])
        seg_candidates = current_config.get('candidates', {}).get('segment', 20)
        start_candidates = current_config.get('candidates', {}).get('start', 100)
        feedrate = plotter_cfg.get("feedrate", 300)
        
        # Clone canvas for physical plotting
        plotting_canvas = current_canvas.copy()
        
        total_segments = 0
        max_segments = 1000000  # Increased from 1000 to 1000000
        
        # Main plotting loop
        while plotting_active and total_segments < max_segments:
            # Choose new starting point
            start_x, start_y = select_next_start(plotting_canvas, current_target_gray, num_candidates=start_candidates)
            current_theta = np.random.uniform(0, 2*np.pi)
            current_x, current_y = start_x, start_y
            scribble_color = tuple(int(c) for c in colors[0])  # Use first color for now
            
            # Transform start position
            plot_start = transformer.transform(start_x, start_y)
            
            scribble_start_error = compute_error(plotting_canvas, current_target_gray)
            scribble_count = 0
            first_segment = True
            
            while plotting_active and scribble_count < scribble_max:
                current_error = compute_error(plotting_canvas, current_target_gray)
                best_error = None
                best_canvas = None
                best_endpoint = (None, None)
                best_new_theta = None
                best_arc = None
                
                for _ in range(seg_candidates):
                    L = np.random.uniform(seg_length_min, seg_length_max)
                    delta = np.random.uniform(delta_min, delta_max)
                    candidate_canvas, candidate_x, candidate_y, candidate_theta, arc_info = draw_curve(
                        plotting_canvas, current_x, current_y, current_theta, L, delta, scribble_color, stroke_thickness
                    )
                    err = compute_error(candidate_canvas, current_target_gray)
                    
                    if best_error is None or err < best_error:
                        best_error = err
                        best_canvas = candidate_canvas
                        best_endpoint = (candidate_x, candidate_y)
                        best_new_theta = candidate_theta
                        best_arc = arc_info
                
                # If no improvement found, skip this scribble
                if best_error is None or best_error >= current_error:
                    break
                
                # Only move to start position and lower pen on first segment
                if first_segment:
                    plotter.move_linear(plot_start[0], plot_start[1], feedrate)
                    plotter.lower_pen()
                    first_segment = False
                
                plotting_canvas = best_canvas
                prev_x, prev_y = current_x, current_y
                current_x, current_y = best_endpoint
                current_theta = best_new_theta
                
                # Plot segment
                plot_prev = transformer.transform(prev_x, prev_y)
                plot_current = transformer.transform(current_x, current_y)
                
                if best_arc is not None:
                    radius = best_arc["r"] * transformer.scale  # Scale the radius
                    clockwise = best_arc["delta"] > 0
                    # Transform to plotter space and invert clockwise due to Y inversion
                    plotter.move_arc(plot_prev[0], plot_prev[1], 
                                   plot_current[0], plot_current[1], 
                                   radius, not clockwise)
                else:
                    plotter.move_linear(plot_current[0], plot_current[1], feedrate)
                
                scribble_count += 1
                total_segments += 1
                
                # Update progress
                progress['total_segments'] = total_segments
                progress['current_error'] = int(best_error)
                progress['percent_complete'] = min(100, int(total_segments / max_segments * 100))
                
                # Check for stopping conditions
                new_err = compute_error(plotting_canvas, current_target_gray)
                improvement = scribble_start_error - new_err
                if scribble_count >= scribble_min and improvement < scribble_start_error * improvement_threshold:
                    break
            
            # Lift pen after each scribble
            if not first_segment:  # Only if we actually drew something
                plotter.lift_pen()
        
        # Close plotter connection
        plotter.close()
        progress['status'] = 'completed'
        
    except Exception as e:
        progress['status'] = 'error'
        progress['error'] = str(e)
        
        # Try to ensure the pen is lifted and connection closed if there's an error
        try:
            if 'plotter' in locals():
                plotter.lift_pen()
                plotter.close()
        except:
            pass
    
    plotting_active = False

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
