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
        
        points = list(points or [])
        x, y = evt.index[0], evt.index[1]
        label = 1 if mode == "Foreground" else 0
        points = points + [(x, y, label)]
        
        # Visualize points
        img_points = self.visualize_points(img, points)
        return points, img_points

    def remove_point(self, img, points, evt: gr.SelectData):
        # Click on a point to remove it
        if not points or img is None:
            return points, img
        
        points = list(points)
        x, y = evt.index[0], evt.index[1]
        tolerance = 10
        
        # Find and remove closest point
        for i, (px, py, label) in enumerate(points):
            if abs(x - px) < tolerance and abs(y - py) < tolerance:
                points = points[:i] + points[i+1:]
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
        radius = 10
        
        for x, y, label in points:
            color = self.foreground_color if label == 1 else self.background_color
            color = tuple(color.tolist())
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], outline=color, fill=color, width=2)
        
        return img
    
    def visualize_mask(self, img, mask):
        # Overlay mask on image for visualization
        if img is None:
            return img
        if mask is None:
            return img
        
        img = np.array(img.copy())
        alpha = 0.6
        img[mask > 0.5] = (alpha * self.foreground_color + (1 - alpha) * img[mask > 0.5]).astype(np.uint8)
        img[mask <= 0.5] = (alpha * self.background_color + (1 - alpha) * img[mask <= 0.5]).astype(np.uint8)
        
        return Image.fromarray(img)
    
    def visualize_style_transfer(self, img_styled1, img_styled2, mask):
        # Mix two styled results by mask. Works with tensors or PIL images.
        def to_float_array(img):
            if img is None:
                return None
            if hasattr(img, 'cpu'):
                arr = img.cpu().numpy()
                # expected tensor shape (C,H,W)
                if arr.ndim == 4:
                    arr = arr[0]
                arr = arr.transpose(1, 2, 0)
                return arr.astype(np.float32)
            # PIL Image
            arr = np.array(img).astype(np.float32) / 255.0
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            return arr

        def to_pil(arr):
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            return Image.fromarray(arr)

        if img_styled1 is None and img_styled2 is None:
            return None

        if mask is None:
            # just return first available styled output
            styled = img_styled1 if img_styled1 is not None else img_styled2
            if styled is None:
                return None
            if hasattr(styled, 'cpu'):
                return to_pil(to_float_array(styled))
            return styled

        # mask exists: blend
        mask_arr = np.array(mask)
        if mask_arr.ndim == 3 and mask_arr.shape[2] == 1:
            mask_arr = mask_arr[..., 0]
        if mask_arr.ndim == 3:
            # if mask has 3 channels, reduce to one
            mask_arr = mask_arr[..., 0]

        mask_arr = mask_arr.astype(np.float32)
        if mask_arr.max() > 1.0:
            mask_arr = mask_arr / 255.0
        mask_arr = mask_arr.clip(0.0, 1.0)
        mask_arr = mask_arr[..., None]  # (H,W,1)

        arr1 = to_float_array(img_styled1) if img_styled1 is not None else None
        arr2 = to_float_array(img_styled2) if img_styled2 is not None else None

        if arr1 is None:
            arr1 = arr2
        if arr2 is None:
            arr2 = arr1

        if arr1.shape[:2] != mask_arr.shape[:2]:
            # fall back to returning the first style if sizes mismatch
            return to_pil(arr1)

        blended = arr1 * mask_arr + arr2 * (1.0 - mask_arr)
        return to_pil(blended)


    def toggle_mode(self, mode):
        # Toggle between Foreground and Background mode
        new_mode = "Background" if mode == "Foreground" else "Foreground"
        return new_mode, f"Mode: {new_mode} (click to toggle)"

    def reset_points(self, img):
        # Clear all points and reset outputs
        return [], img, img, img