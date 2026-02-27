import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

class Interface:
    def __init__(self):
        self.foreground_color = np.array([255, 255, 0])
        self.background_color = np.array([0, 0, 255])

    def add_point(self, img, points, mode, evt: gr.SelectData):
        # Add a point to the list and update visualization
        if img is None:
            return points, img
        
        points = points or []
        x, y = evt.index[0], evt.index[1]
        label = 1 if mode == "Foreground" else 0
        points.append((x, y, label))
        
        # Visualize points
        img_points = self.visualize_points(img, points)
        return points, img_points

    def remove_point(self, img, points, evt: gr.SelectData):
        # Click on a point to remove it
        if not points or img is None:
            return points, img
        
        x, y = evt.index[0], evt.index[1]
        tolerance = 10
        
        # Find and remove closest point
        for i, (px, py, label) in enumerate(points):
            if abs(x - px) < tolerance and abs(y - py) < tolerance:
                points.pop(i)
                break
        
        # Update visualization
        img_points = self.visualize_points(img, points)
        return points, img_points

    def visualize_points(self, img, points):
        # Draw circles on image to show clicked points
        if img is None or not points:
            return img
        
        img = img.copy()
        draw = ImageDraw.Draw(img)
        radius = 20
        
        for x, y, label in points:
            color = self.foreground_color if label == 1 else self.background_color
            color = tuple(color.tolist())
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], outline=color, fill=color, width=2)
        
        return img
    
    def visualize_mask(self, img, mask):
        # Overlay mask on image for visualization
        if img is None or mask is None:
            return img
        
        img = np.array(img.copy())
        alpha = 0.6
        img[mask > 0.5] = (alpha * self.foreground_color + (1 - alpha) * img[mask > 0.5]).astype(np.uint8)
        img[mask <= 0.5] = (alpha * self.background_color + (1 - alpha) * img[mask <= 0.5]).astype(np.uint8)
        
        return Image.fromarray(img)
    
    def visualize_style_transfer(self, img, stylized):
        # TODO: Implement style transfer visualization
        pass

    def toggle_mode(self, mode):
        # Toggle between Foreground and Background mode
        new_mode = "Background" if mode == "Foreground" else "Foreground"
        return new_mode, f"Mode: {new_mode} (click to toggle)"

    def reset_all(self, img):
        """Clear all points and reset outputs."""
        return [], img, img, img