from abc import ABC, abstractmethod
import time
import glob
import cv2
import numpy as np
import math
import random
import argparse
import yaml
import os
import datetime
import serial 


# Try to import pyserial
try:
    import serial
except ImportError:
    serial = None

class PlotterInterface(ABC):
    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def move_linear(self, x, y, feedrate):
        pass

    @abstractmethod
    def move_arc(self, x_start, y_start, x_end, y_end, I, J, feedrate, clockwise):
        pass

    @abstractmethod
    def lift_pen(self):
        pass

    @abstractmethod
    def lower_pen(self):
        pass

    @abstractmethod
    def close(self):
        pass

class PlotterBase(PlotterInterface, ABC):
    """
    A base class for plotters that implements all common G-code operations.
    Subclasses only need to override the _write() method.
    """
    def __init__(self, plotter_config):
        self.config = plotter_config
        self.feedrate = plotter_config.get("feedrate", 300)

    @abstractmethod
    def _write(self, cmd, wait_for_ok=True):
        """
        Write a G-code command.
        Subclasses implement this to actually output the command (e.g. to serial or a file).
        If wait_for_ok is True, the method should block until an acknowledgment is received.
        """
        pass

    def initialize(self):
        # Wake GRBL (or write header commands)
        self._write("\r\n\r\n", wait_for_ok=False)
        time.sleep(2)
        self._write("$X\n")
        self._write("G21\n")  # Set units to mm
        self._write("G90\n")  # Absolute positioning
        self._write("$H\n")   # Home
        time.sleep(5)
        # Set coordinate system via G92.
        g92_vals = self.config.get("g92", [0, 0, 0])
        self._write(f"G92 X{g92_vals[0]} Y{g92_vals[1]} Z{g92_vals[2]}\n")

    def move_linear(self, x, y, feedrate):
        cmd = f"G1 X{x:.5f} Y{y:.5f} F{feedrate}\n"
        self._write(cmd)

    def move_arc(self, x_start, y_start, x_end, y_end, radius, clockwise):
        """Generate G2/G3 command using R parameter instead of I,J"""
        cmd_type = "G3" if clockwise else "G2"
        cmd = f"{cmd_type} X{x_end:.5f} Y{y_end:.5f} R{radius:.5f} F{self.feedrate}\n"
        self._write(cmd)

    def lift_pen(self):
        z_up = self.config.get("z_up", 5)
        cmd = f"G1 Z{z_up:.5f} F{self.feedrate}\n"
        self._write(cmd)

    def lower_pen(self):
        z_down = self.config.get("z_down", 0)
        cmd = f"G1 Z{z_down:.5f} F{self.feedrate}\n"
        self._write(cmd)

    def close(self):
        # By default, do nothing here; subclasses can override if needed.
        pass

class SerialPlotter(PlotterBase):
    def __init__(self, plotter_config):
        if serial is None:
            raise RuntimeError("pyserial is not installed.")
        port = plotter_config.get("serial_port", "")
        if not port:
            ports = glob.glob("/dev/cu.usb*")
            if ports:
                port = ports[0]
            else:
                raise RuntimeError("No serial port found matching /dev/cu.usb*")
        self.port = port
        self.baudrate = plotter_config.get("baudrate", 115200)
        try:
            self.ser = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=1)
            time.sleep(2)
            self.ser.flushInput()
        except Exception as e:
            raise RuntimeError(f"Failed to open serial port {self.port}: {e}")
        super().__init__(plotter_config)

    def _write(self, cmd, wait_for_ok=True):
        print("Sending:", cmd.strip())
        self.ser.write(cmd.encode())
        self.ser.flush()
        if wait_for_ok:
            while True:
                line = self.ser.readline().decode('utf-8').strip()
                if line:
                    print("Received:", line)
                    if "ok" in line.lower() or "error" in line.lower():
                        break

    def close(self):
        if self.ser:
            self.ser.close()

class FilePlotter(PlotterBase):
    def __init__(self, plotter_config, filename="output.gcode"):
        self.file = open(filename, "w")
        print("G-code will be written to", filename)
        super().__init__(plotter_config)

    def _write(self, cmd, wait_for_ok=True):
        print("Writing:", cmd.strip())
        self.file.write(cmd)

    def close(self):
        self.file.close()


from abc import ABC, abstractmethod
import time
import glob

# Assuming pyserial is already imported if available.
try:
    import serial
except ImportError:
    serial = None

class PlotterInterface(ABC):
    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def move_linear(self, x, y, feedrate):
        pass

    @abstractmethod
    def move_arc(self, x_start, y_start, x_end, y_end, I, J, feedrate, clockwise):
        pass

    @abstractmethod
    def lift_pen(self):
        pass

    @abstractmethod
    def lower_pen(self):
        pass

    @abstractmethod
    def close(self):
        pass

class PlotterBase(PlotterInterface, ABC):
    """
    A base class for plotters. It implements initialize(), move_linear(), move_arc(),
    lift_pen(), and lower_pen() in terms of a low-level _write(cmd) method, which the
    subclasses must implement.
    """
    def __init__(self, plotter_config):
        self.config = plotter_config
        self.feedrate = plotter_config.get("feedrate", 300)

    @abstractmethod
    def _write(self, cmd, wait_for_ok=True):
        """
        Write a command string to the output (serial port or file).
        If wait_for_ok is True, wait for a response.
        """
        pass

    def initialize(self):
        # Wake GRBL (or write header commands)
        self._write("\r\n\r\n", wait_for_ok=False)
        time.sleep(2)
        self._write("$X\n")      # Unlock
        self._write("G21\n")     # Set units to mm
        self._write("G90\n")     # Absolute positioning
        self._write("$H\n")      # Home
        time.sleep(5)
        # Set coordinate system using G92.
        g92_vals = self.config.get("g92", [0, 0, 0])
        self._write(f"G92 X{g92_vals[0]} Y{g92_vals[1]} Z{g92_vals[2]}\n")

    def move_linear(self, x, y, feedrate):
        cmd = f"G1 X{x:.5f} Y{y:.5f} F{feedrate}\n"
        self._write(cmd)

    def move_arc(self, x_start, y_start, x_end, y_end, radius, clockwise):
        """Generate G2/G3 command using R parameter instead of I,J"""
        cmd_type = "G3" if clockwise else "G2"
        cmd = f"{cmd_type} X{x_end:.5f} Y{y_end:.5f} R{radius:.5f} F{self.feedrate}\n"
        self._write(cmd)

    def lift_pen(self):
        print("Lifting pen.")
        z_up = self.config.get("z_up", 5)
        cmd = f"G1 Z{z_up:.5f} F{self.feedrate}\n"
        self._write(cmd)

    def lower_pen(self):
        print("Lowering pen.")
        z_down = self.config.get("z_down", 0)
        cmd = f"G1 Z{z_down:.5f} F{self.feedrate}\n"
        self._write(cmd)

    def close(self):
        # Base class does nothing here.
        pass

class SerialPlotter(PlotterBase):
    def __init__(self, plotter_config):
        if serial is None:
            raise RuntimeError("pyserial not installed.")
        # If no serial port is provided, search for /dev/cu.usb* (e.g., on macOS)
        port = plotter_config.get("serial_port", "")
        if not port:
            ports = glob.glob("/dev/cu.usb*")
            if ports:
                port = ports[0]
            else:
                raise RuntimeError("No serial port found matching /dev/cu.usb*")
        self.port = port
        self.baudrate = plotter_config.get("baudrate", 115200)
        try:
            self.ser = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=1)
            time.sleep(2)
            self.ser.flushInput()
        except Exception as e:
            raise RuntimeError(f"Failed to open serial port {self.port}: {e}")
        super().__init__(plotter_config)

    def _write(self, cmd, wait_for_ok=True):
        print("Sending:", cmd.strip())
        self.ser.write(cmd.encode())
        self.ser.flush()
        if wait_for_ok:
            while True:
                line = self.ser.readline().decode('utf-8').strip()
                if line:
                    print("Received:", line)
                    if "ok" in line.lower() or "error" in line.lower():
                        break

    def close(self):
        if self.ser:
            self.ser.close()

class FilePlotter(PlotterBase):
    def __init__(self, plotter_config, filename="output.gcode"):
        self.file = open(filename, "w")
        print("G-code will be written to", filename)
        super().__init__(plotter_config)

    def _write(self, cmd, wait_for_ok=True):
        print("Writing:", cmd.strip())
        self.file.write(cmd)

    def close(self):
        self.file.close()


#######################################
# Coordinate Transformation Class
#######################################

class CoordinateTransformer:
    def __init__(self, canvas_width, canvas_height, plotter_config):
        """Initialize coordinate transformer that preserves aspect ratio."""
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        
        # Get plotter bounds
        self.x_min = plotter_config.get("x_min", 0)
        self.x_max = plotter_config.get("x_max", 100)
        self.y_min = plotter_config.get("y_min", 0)
        self.y_max = plotter_config.get("y_max", 100)
        
        # Calculate available plotter area
        plotter_width = self.x_max - self.x_min
        plotter_height = self.y_max - self.y_min
        
        # Calculate scale factor preserving aspect ratio
        canvas_aspect = canvas_width / canvas_height
        plotter_aspect = plotter_width / plotter_height
        
        if canvas_aspect > plotter_aspect:
            # Canvas is wider relative to height - fit to width
            self.scale = plotter_width / canvas_width
            scaled_height = canvas_height * self.scale
            # Center vertically
            self.y_offset = (plotter_height - scaled_height) / 2
            self.x_offset = 0
        else:
            # Canvas is taller relative to width - fit to height
            self.scale = plotter_height / canvas_height
            scaled_width = canvas_width * self.scale
            # Center horizontally
            self.x_offset = (plotter_width - scaled_width) / 2
            self.y_offset = 0

    def transform(self, x, y):
        """
        Convert canvas coordinates to plotter coordinates preserving aspect ratio.
        Canvas: origin at top-left, Y grows down
        Plotter: origin at bottom-left, Y grows up
        """
        # Scale X normally but invert Y (canvas Y=0 maps to max plotter Y)
        X = x * self.scale + self.x_min + self.x_offset
        Y = self.y_max - (y * self.scale + self.y_offset)  # Invert Y axis
        return X, Y

    def transform_arc(self, x_start, y_start, x_end, y_end, I, J, clockwise):
        """
        Transform arc parameters and adjust clockwise flag.
        When Y is inverted, clockwise in canvas becomes counterclockwise in plotter space.
        
        Args:
            x_start, y_start: Arc start point in canvas coordinates
            x_end, y_end: Arc end point in canvas coordinates
            I, J: Center point offsets from start point in canvas coordinates
            clockwise: Arc direction in canvas coordinates
        Returns:
            Transformed start, end coordinates and I,J offsets for plotter space
        """
        # Transform start and end points
        start_x, start_y = self.transform(x_start, y_start)
        end_x, end_y = self.transform(x_end, y_end)
        
        # Transform the center point (which is offset from start by I,J)
        center_x, center_y = self.transform(x_start + I, y_start + J)
        
        # Calculate new I,J as offsets from transformed start point to transformed center
        new_I = center_x - start_x
        new_J = center_y - start_y
        
        # Invert clockwise flag due to Y axis inversion
        new_clockwise = not clockwise
        
        return start_x, start_y, end_x, end_y, new_I, new_J, new_clockwise

#######################################
# Video Recorder Class
#######################################

class VideoRecorder:
    def __init__(self, canvas_width, canvas_height, base_filename, fps=30):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base_filename}_{date_str}.mp4"
        self.writer = cv2.VideoWriter(filename, fourcc, fps, (canvas_width, canvas_height))
        if self.writer.isOpened():
            print("Recording video to:", filename)
        else:
            print("Error: VideoWriter failed to open.")
            self.writer = None

    def record(self, frame):
        if self.writer:
            self.writer.write(frame)

    def release(self):
        if self.writer:
            self.writer.release()

#######################################
# Utility Functions
#######################################

def fit_image_on_canvas(target_img, canvas_width, canvas_height):
    """Scale target_img (grayscale) preserving aspect ratio and center it on a white background."""
    h, w = target_img.shape[:2]
    scale = min(canvas_width / w, canvas_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(target_img, (new_w, new_h))
    canvas_img = np.full((canvas_height, canvas_width), 255, dtype=np.uint8)
    x_offset = (canvas_width - new_w) // 2
    y_offset = (canvas_height - new_h) // 2
    canvas_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas_img

def draw_curve(canvas, x0, y0, theta, L, delta, color, thickness):
    """
    Draw a curve segment (or a straight line if delta is nearly zero) onto a temporary stroke layer,
    blend it with the canvas (via per-pixel minimum), and return the new canvas along with the
    endpoint (x1, y1), new tangent (theta+delta) and arc_info (if an arc was drawn, else None).
    """
    stroke_layer = np.full(canvas.shape, 255, dtype=np.uint8)
    arc_info = None
    if abs(delta) < 1e-6:
        x1 = x0 + L * math.cos(theta)
        y1 = y0 + L * math.sin(theta)
        pt0 = (int(round(x0)), int(round(y0)))
        pt1 = (int(round(x1)), int(round(y1)))
        cv2.line(stroke_layer, pt0, pt1, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    else:
        r = L / abs(delta)
        if delta > 0:
            cx = x0 - r * math.sin(theta)
            cy = y0 + r * math.cos(theta)
        else:
            cx = x0 + r * math.sin(theta)
            cy = y0 - r * math.cos(theta)
        arc_info = {"cx": cx, "cy": cy, "r": r, "delta": delta}
        sign = 1 if delta > 0 else -1
        start_angle = theta - sign * math.pi / 2
        end_angle = start_angle + delta
        
        # Calculate exact endpoint before rounding
        x1 = cx + r * math.cos(end_angle)
        y1 = cy + r * math.sin(end_angle)
        
        # Assert using exact endpoint coordinates
        assert(abs(math.hypot(x1 - cx, y1 - cy) - r) < 1e-6)

        # Draw the polyline with rounded points for display
        num_points = max(int(L), 2)
        angles = np.linspace(start_angle, end_angle, num_points)
        pts = []
        for a in angles:
            x = cx + r * math.cos(a)
            y = cy + r * math.sin(a)
            pts.append([int(round(x)), int(round(y))])
        pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(stroke_layer, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        
        # Use exact endpoint coordinates for return value instead of rounded point

    new_canvas = np.minimum(canvas, stroke_layer)
    new_theta = theta + delta
    return new_canvas, x1, y1, new_theta, arc_info

def compute_error(canvas, target_gray):
    """Compute error as the sum of squared differences of Gaussian-blurred images."""
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    blurred_canvas = cv2.GaussianBlur(canvas_gray, (5, 5), 0)
    blurred_target = cv2.GaussianBlur(target_gray, (1, 1), 0)
    diff = blurred_canvas.astype(np.int32) - blurred_target.astype(np.int32)
    return np.sum(diff * diff)

def select_next_start(canvas, target_gray, num_candidates):
    """Select the canvas coordinate with the highest local error (after blurring) from a random sample."""
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    blurred_canvas = cv2.GaussianBlur(canvas_gray, (3, 3), 0)
    blurred_target = cv2.GaussianBlur(target_gray, (3, 3), 0)
    error_map = np.abs(blurred_canvas.astype(np.int32) - blurred_target.astype(np.int32))
    best_error = -1
    best_coord = (None, None)
    h, w = error_map.shape
    for _ in range(num_candidates):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        if error_map[y, x] > best_error:
            best_error = error_map[y, x]
            best_coord = (x, y)
    return float(best_coord[0]), float(best_coord[1])

#######################################
# Main Program
#######################################

def main():
    parser = argparse.ArgumentParser(
        description="Plotter Scribble Simulation with G-code generation (G2/G3 arcs) and encapsulated plotting and video-recording."
    )
    parser.add_argument("image", help="Path to the target image (JPG)")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML configuration file")
    parser.add_argument("--record", action="store_true", help="Generate MP4 video recording of the canvas")
    parser.add_argument("--plotter", choices=["serial", "file", "none"], default="none",
                        help="Plotter mode: 'serial' to send to a GRBL plotter, 'file' to dump G-code, 'none' to disable plotting")
    parser.add_argument("--gcodefile", default="output.gcode", help="Output file for G-code if using file plotter")
    args = parser.parse_args()

    print("Working directory:", os.getcwd())
    config = yaml.safe_load(open(args.config, "r"))

    # Canvas configuration.
    canvas_cfg = config.get("canvas", {})
    canvas_width = canvas_cfg.get("width")
    canvas_height = canvas_cfg.get("height")
    
    if not canvas_width or not canvas_height:
        print("Error: Canvas width and height must be specified in config file")
        return

    # Scribble and segment parameters.
    scribble_min = config['scribble'].get('min_segments', 5)
    scribble_max = config['scribble'].get('max_segments', 200)
    improvement_threshold = config['scribble'].get('improvement_threshold', 0.01)
    seg_length_min = config['segment'].get('length_min', 5)
    seg_length_max = config['segment'].get('length_max', 30)
    delta_min = config['segment'].get('delta_min', -math.pi/2)
    delta_max = config['segment'].get('delta_max', math.pi/2)
    stroke_thickness = config['stroke'].get('thickness', 3)
    colors = config.get('colors', [[0, 0, 0]])
    seg_candidates = config.get('candidates', {}).get('segment', 20)
    start_candidates = config.get('candidates', {}).get('start', 100)

    plotter_cfg = config.get("plotter", {})
    feedrate = plotter_cfg.get("feedrate", 300)

    # Instantiate plotter if needed.
    plotter = None
    if args.plotter == "serial":
        try:
            plotter = SerialPlotter(plotter_cfg)
        except Exception as e:
            print("Error initializing SerialPlotter:", e)
            return
    elif args.plotter == "file":
        plotter = FilePlotter(plotter_cfg, filename=args.gcodefile)

    if plotter is not None:
        plotter.initialize()

    # Instantiate coordinate transformer.
    transformer = CoordinateTransformer(canvas_width, canvas_height, plotter_cfg)

    # Instantiate video recorder if needed.
    video_recorder = None
    if args.record:
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        video_recorder = VideoRecorder(canvas_width, canvas_height, base_name, fps=30)

    # Load and fit target image.
    target_img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if target_img is None:
        print("Error loading image.")
        return
    target_gray = fit_image_on_canvas(target_img, canvas_width, canvas_height)

    # Create white canvas (color).
    canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)
    cv2.namedWindow("Canvas", cv2.WINDOW_NORMAL)

    total_segments = 0

    while True:
        # Choose new starting point.
        start_x, start_y = select_next_start(canvas, target_gray, num_candidates=start_candidates)
        current_theta = random.uniform(0, 2*math.pi)
        current_x, current_y = start_x, start_y
        scribble_color = tuple(int(c) for c in random.choice(colors))

        # Calculate transformed start position but don't move there yet
        if plotter is not None:
            plot_start = transformer.transform(start_x, start_y)

        scribble_start_error = compute_error(canvas, target_gray)
        scribble_count = 0
     
        while True:
            current_error = compute_error(canvas, target_gray)
            best_error = None
            best_canvas = None
            best_endpoint = (None, None)
            best_new_theta = None
            best_arc = None
            for _ in range(seg_candidates):
                L = random.uniform(seg_length_min, seg_length_max)
                delta = random.uniform(delta_min, delta_max)
                candidate_canvas, candidate_x, candidate_y, candidate_new_theta, arc_info = draw_curve(
                    canvas, current_x, current_y, current_theta, L, delta, scribble_color, stroke_thickness)
                err = compute_error(candidate_canvas, target_gray)
                if best_error is None or err < best_error:
                    best_error = err
                    best_canvas = candidate_canvas
                    best_endpoint = (candidate_x, candidate_y)
                    best_new_theta = candidate_new_theta
                    best_arc = arc_info

            # If no improvement found, skip this scribble entirely
            if best_error >= current_error:
                print(f"Segment {total_segments:06d}: No improvement. Skipping scribble.")
                break

            # Only when first successful segment is found:
            # 1. Move to start position
            # 2. Lower pen
            if plotter is not None and scribble_count == 0:
                print(f"First successful segment found - moving to start position and lowering pen")
                plotter.move_linear(plot_start[0], plot_start[1], feedrate)
                plotter.lower_pen()

            canvas = best_canvas
            prev_x, prev_y = current_x, current_y
            current_x, current_y = best_endpoint
            current_theta = best_new_theta
            scribble_count += 1
            total_segments += 1

            cv2.imshow("Canvas", canvas)
            if video_recorder is not None:
                video_recorder.record(canvas)
            print(f"Segment {total_segments:06d}, Scribble segments: {scribble_count}, Error: {best_error}")

            if plotter is not None:
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

            # Instead of lifting the pen immediately when Esc is pressed,
            # simply break out of the inner loop so that the pen is lifted only at the end of the scribble.
            if cv2.waitKey(1) & 0xFF == 27:
                print("Esc pressed. Ending scribble.")
                break

            new_err = compute_error(canvas, target_gray)
            improvement = scribble_start_error - new_err
            if scribble_count >= scribble_min and improvement < scribble_start_error * improvement_threshold:
                print("Ending scribble due to insufficient improvement.")
                break
            if scribble_count >= scribble_max:
                print("Ending scribble due to maximum segments reached.")
                break
        # Lift pen only once after the current scribble is finished.
        if plotter is not None:
            print("end of scribble reached - lifting pen")
            plotter.lift_pen()
            
    # (Optional) Continue with the next scribble.

    if plotter is not None:
        plotter.close()
    if video_recorder is not None:
        video_recorder.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()