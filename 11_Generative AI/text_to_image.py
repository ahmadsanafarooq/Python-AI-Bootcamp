import gradio as gr
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import spaces

# Initialize the model
MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pipeline
@spaces.GPU
def load_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(DEVICE)
    
    if torch.cuda.is_available():
        pipe.enable_memory_efficient_attention()
        pipe.enable_xformers_memory_efficient_attention()
    
    return pipe

# Load pipeline at startup
pipe = load_pipeline()

@spaces.GPU
def generate_image(prompt, negative_prompt="", num_inference_steps=25, guidance_scale=7.5, width=512, height=512, seed=-1):
    """
    Generate image from text prompt using Stable Diffusion
    """
    try:
        # Set seed for reproducibility
        if seed != -1:
            generator = torch.Generator(device=DEVICE).manual_seed(seed)
        else:
            generator = None
        
        # Generate image
        with torch.autocast(DEVICE):
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator
            ).images[0]
        
        return image
    
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Text-to-Image Generator", theme=gr.themes.Soft()) as demo:
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h1>🎨 Text-to-Image Generator</h1>
            <p>Generate beautiful images from text descriptions using Stable Diffusion</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input controls
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your image description here...",
                    lines=3,
                    value="A beautiful sunset over mountains, highly detailed, 8k resolution"
                )
                
                negative_prompt_input = gr.Textbox(
                    label="Negative Prompt (Optional)",
                    placeholder="What you don't want in the image...",
                    lines=2,
                    value="blurry, low quality, distorted"
                )
                
                with gr.Row():
                    steps_slider = gr.Slider(
                        minimum=10,
                        maximum=50,
                        value=25,
                        step=1,
                        label="Inference Steps"
                    )
                    
                    guidance_slider = gr.Slider(
                        minimum=1.0,
                        maximum=20.0,
                        value=7.5,
                        step=0.5,
                        label="Guidance Scale"
                    )
                
                with gr.Row():
                    width_slider = gr.Slider(
                        minimum=256,
                        maximum=1024,
                        value=512,
                        step=64,
                        label="Width"
                    )
                    
                    height_slider = gr.Slider(
                        minimum=256,
                        maximum=1024,
                        value=512,
                        step=64,
                        label="Height"
                    )
                
                seed_input = gr.Number(
                    label="Seed (-1 for random)",
                    value=-1,
                    precision=0
                )
                
                generate_btn = gr.Button(
                    "🎨 Generate Image",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                # Output
                output_image = gr.Image(
                    label="Generated Image",
                    type="pil",
                    height=500
                )
        
        # Example prompts
        gr.HTML("<h3>💡 Example Prompts</h3>")
        examples = gr.Examples(
            examples=[
                ["A majestic lion in a savanna at golden hour, photorealistic, 8k"],
                ["Cyberpunk city at night with neon lights, futuristic, detailed"],
                ["A cozy cottage in a magical forest, fairy tale style, warm lighting"],
                ["Abstract art with vibrant colors, geometric shapes, modern"],
                ["A space explorer on an alien planet, sci-fi, dramatic lighting"],
            ],
            inputs=prompt_input,
            label="Click on an example to try it out!"
        )
        
        # Event handlers
        generate_btn.click(
            fn=generate_image,
            inputs=[
                prompt_input,
                negative_prompt_input,
                steps_slider,
                guidance_slider,
                width_slider,
                height_slider,
                seed_input
            ],
            outputs=output_image,
            show_progress=True
        )
        
        # Allow Enter key to generate
        prompt_input.submit(
            fn=generate_image,
            inputs=[
                prompt_input,
                negative_prompt_input,
                steps_slider,
                guidance_slider,
                width_slider,
                height_slider,
                seed_input
            ],
            outputs=output_image,
            show_progress=True
        )
    
    return demo

# Launch the app
if __name__ == "__main__":
    demo = create_interface()
    demo.queue(max_size=20)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )