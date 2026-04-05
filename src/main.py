import gradio as gr
import numpy as np
from PIL import Image

from segment import Segment_model
from interface import Interface
from IPAdapter import IPAdapter
from diffusion import LatentCompositeDiffusion

class Controller:
    def __init__(self, segmenter, interface, style_model, diffusion_model):
        self.segmenter = segmenter
        self.interface = interface
        self.style_model = style_model
        self.diffusion_model = diffusion_model

    def style_transfer(self, img, style1, style2):
        img_styled1 = self.style_model.stylize(img, style1)
        img_styled2 = self.style_model.stylize(img, style2)
        return img_styled1, img_styled2

    def segment_img(self, img, points):
        # Run segmentation with the current points and labels
        if img is None or not points:
            return None
        
        coords = [[p[0], p[1]] for p in points]
        labels = [p[2] for p in points]
        
        return self.segmenter.segment(img, coords, labels)
    
    def refine_diffusion(self, fg_img, bg_img, mask, prompt=""):
        self.diffusion_model.train(fg_img, bg_img, mask, prompt=prompt)
        return self.diffusion_model.generate()

    def update(self, img, style1, style2, points, mode, action, evt: gr.SelectData):
        # Update interface points, then segment, then return updated values
        if action == "add":
            points, img_points = self.interface.add_point(img, points, mode, evt)
        elif action == "remove":
            points, img_points = self.interface.remove_point(img, points, evt)
        else:
            return points, img, img, img

        mask = self.segment_img(img, points)
        img_segmented = self.interface.visualize_mask(img, mask)

        # style transfer only if both style images exist
        if style1 is not None and style2 is not None:
            img_styled1, img_styled2 = self.style_transfer(img, style1, style2)
            img_styled = self.interface.visualize_style_transfer(img_styled1, img_styled2, mask)
        else:
            # fall back to showing original image when no styles
            img_styled = img

        img_refined = self.refine_diffusion(img_styled1, img_styled2, mask)
        
        return points, img_points, img_segmented, img_styled1, img_styled2, img_styled, img_refined
    
    def preprocess(self, img):
        # Resize image to 512x512 for model input
        if img is None:
            return None
        return img.resize((512, 512), Image.BILINEAR)
    
if __name__ == "__main__":
    segmenter = Segment_model()
    interface = Interface()
    ip_adapter_model = IPAdapter()
    diffusion_model = LatentCompositeDiffusion()

    controller = Controller(segmenter, interface, ip_adapter_model, diffusion_model)

    with gr.Blocks() as demo:
        points_state = gr.State([])
        mode_state = gr.State("Foreground")

        with gr.Tabs():
            with gr.TabItem("Main"):
                # Segmentation Editing Inputs
                with gr.Row():
                    img_input = gr.Image(type="pil", interactive=True, label="Click to add points")
                    img_points = gr.Image(type="pil", label="Points (click to remove)")

                # Segmentation Editing Outputs
                with gr.Row():
                    img_segmentation = gr.Image(type="pil", label="Segmentation")
                    img_style_seg = gr.Image(type="pil", label="Style Transfer Segmented")

                with gr.Row():
                    img_style1 = gr.Image(type="pil", label="Style Transfer 1")
                    img_style2 = gr.Image(type="pil", label="Style Transfer 2")

                # Buttons for Segmentation
                with gr.Row():
                    btn_toggle = gr.Button("Mode: Foreground (click to toggle)")
                    btn_reset = gr.Button("Reset Points")

                img_output_3 = gr.Image(type="pil", label="Diffusion Style Transfer")


            with gr.TabItem("Styles"):
                # Style Transfer styles
                img_style_1 = gr.Image(type="pil", label="Style 1")
                img_style_2 = gr.Image(type="pil", label="Style 2")
        
        # Event handlers
        img_input.upload(
            controller.preprocess,
            inputs=img_input,
            outputs=img_input
        )
        
        img_input.select(controller.update, 
            inputs=[img_input, img_style_1, img_style_2, points_state, mode_state, gr.State("add")], 
            outputs=[points_state, img_points, img_segmentation, img_style1, img_style2, img_style_seg, img_output_3])
        
        img_points.select(controller.update, 
            inputs=[img_input, img_style_1, img_style_2, points_state, mode_state, gr.State("remove")], 
            outputs=[points_state, img_points, img_segmentation, img_style1, img_style2, img_style_seg, img_output_3])
        
        btn_toggle.click(controller.interface.toggle_mode, 
            inputs=[mode_state], 
            outputs=[mode_state, btn_toggle])
        
        btn_reset.click(controller.interface.reset_points, 
            inputs=[img_input], 
            outputs=[points_state, img_points, img_segmentation, img_style1, img_style2, img_style_seg, img_output_3])

    demo.launch(share=True)