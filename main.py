import gradio as gr
import numpy as np
from PIL import Image

from segment import Segment_model
from interface import Interface

class Controller:
    def __init__(self, segmenter, interface):
        self.segmenter = segmenter
        self.interface = interface

    def style_transfer(self, img, points):
        # TODO: Implement style transfer based on segmented image and points
        pass

    def segment_img(self, img, points):
        # Run segmentation with the current points and labels
        if img is None or not points:
            return img
        
        coords = [[p[0], p[1]] for p in points]
        labels = [p[2] for p in points]
        
        return self.segmenter.segment(img, coords, labels)

    def update(self, img, points, mode, action, evt: gr.SelectData):
        # Update interface points, then segment, then return updated values
        if action == "add":
            points, img_points = self.interface.add_point(img, points, mode, evt)
        elif action == "remove":
            points, img_points = self.interface.remove_point(img, points, evt)
        else:
            return points, img, img, img

        mask = self.segment_img(img, points)
        img_segmented = self.interface.visualize_mask(img, mask)

        return points, img_points, img_segmented, img_segmented

if __name__ == "__main__":
    segmenter = Segment_model()
    interface = Interface()

    controller = Controller(segmenter, interface)

    with gr.Blocks() as demo:
        points_state = gr.State([])
        mode_state = gr.State("Foreground")

        # Images
        with gr.Row():
            img_input = gr.Image(type="pil", interactive=True, label="Click to add points")
            img_points = gr.Image(type="pil", label="Points (click to remove)")
        
        with gr.Row():
            img_output_1 = gr.Image(type="pil", label="Segmentation")
            img_output_2 = gr.Image(type="pil", label="Style Transfer")

        # Buttons
        with gr.Row():
            btn_toggle = gr.Button("Mode: Foreground (click to toggle)")
            btn_reset = gr.Button("Reset Points")

        # Event handlers
        img_input.select(controller.update, 
            inputs=[img_input, points_state, mode_state, gr.State("add")], 
            outputs=[points_state, img_points, img_output_1, img_output_2])
        
        img_points.select(controller.update, 
            inputs=[img_input, points_state, mode_state, gr.State("remove")], 
            outputs=[points_state, img_points, img_output_1, img_output_2])
        
        btn_toggle.click(controller.interface.toggle_mode, 
            inputs=[mode_state], 
            outputs=[mode_state, btn_toggle])
        
        btn_reset.click(controller.interface.reset_all, 
            inputs=[img_input], 
            outputs=[points_state, img_points, img_output_1, img_output_2])
        
        img_input.change(controller.interface.reset_all,
            inputs=[img_input],
            outputs=[points_state, img_points, img_output_1, img_output_2]
    )


    demo.launch(share=False)