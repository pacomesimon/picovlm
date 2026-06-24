import gradio as gr
import pandas as pd
import os
from .core import detect_objects_stream, set_classes_and_save_model
from .core import MODEL_NAMES
from .utils import zip_folder

# Define assets path relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(BASE_DIR, 'assets', 'images')

# Default gallery images
DEFAULT_IMAGES = []
if os.path.exists(IMAGES_DIR):
    # Gradio Gallery with type='filepath' expects a list of [path, label] or [(path, label)]
    # based on the usage in core.py: batch_paths = [img[0] for img in batch]
    DEFAULT_IMAGES = [[os.path.join(IMAGES_DIR, f), None] for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Default prompts 
prompts_df_new = pd.DataFrame({
    "object": ["cardboard box", "packaged product","electronic device"],
    "obstacle": ["person", "machinery", ""],
})

def handle_webcam_change_event(webcam_img, gallery):
    """
    Update gallery with webcam image.
    """
    if webcam_img is None:
        return gallery
    
    # webcam_img is a path if gr.Image(type="filepath")
    new_item = [webcam_img, None]
    
    if gallery is None:
        gallery = []
    
    return gallery + [new_item]

def sync_negative_prompts_columns(pos_df, neg_df):
    """
    Ensure the negative prompts table has exactly the same columns as prompts_table.
    """
    if pos_df is None:
        return pd.DataFrame()
        
    pos_cols = list(pos_df.columns)
    
    if neg_df is None:
        neg_df = pd.DataFrame()
        
    # Rebuild negative dataframe with pos_cols
    # Keep the values if columns match
    new_data = {}
    num_rows = len(neg_df) if len(neg_df) > 0 else 1
    for col in pos_cols:
        if col in neg_df.columns:
            new_data[col] = neg_df[col].tolist()
        else:
            new_data[col] = [""] * num_rows
            
    # Ensure all lists have the same length
    max_len = max(len(v) for v in new_data.values()) if new_data else 1
    for col in pos_cols:
        if len(new_data[col]) < max_len:
            new_data[col] = new_data[col] + [""] * (max_len - len(new_data[col]))
            
    return pd.DataFrame(new_data)

def create_demo():
    
    # Helper for cleanup
    def cleanup_temp_model(file_path):
        if file_path and os.path.exists(file_path):
             try:
                 # Check if it looks like a temp model file just to be safe
                 filename = os.path.basename(file_path)
                 if filename.startswith("yoloe-") and filename.endswith(".pt"):
                    os.remove(file_path)
             except Exception as e:
                 print(f"Error removing file: {e}")

    css = """
    .gradio-container {
        background-image: url('https://live.staticflickr.com/5190/5786786491_2de355c306_b.jpg') !important;
        background-size: cover !important;
        background-position: center !important;
        background-attachment: fixed !important;
    }
    .glass-card {
        background: rgba(15, 23, 42, 0.65) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(107, 114, 128, 0.25) !important;
        border-radius: 5px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5) !important;
        padding: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    .dark .glass-card {
        background: rgba(15, 23, 42, 0.8) !important;
    }
    /* Enhance text readability on dark background */
    .gradio-container h3, .gradio-container p, .gradio-container span {
        text-shadow: 0px 2px 4px rgba(0,0,0,0.5) !important;
    }
    /* Apply glass to common blocks */
    .gr-form, .gr-box, .gr-panel {
        background: rgba(15, 23, 42, 0.4) !important;
        backdrop-filter: blur(8px) !important;
        border: 1px solid rgba(107, 114, 128, 0.2) !important;
    }
    /* Button enhancements */
    button.primary {
        background: #6B7280 !important;
        border: 1px solid rgba(107, 114, 128, 0.4) !important;
        color: #ffffff !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
    }
    button.primary:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 0 15px rgba(107, 114, 128, 0.5) !important;
    }
    """

    with gr.Blocks(title="picovlm - Multi-Modal Labeling", theme=gr.themes.Soft(primary_hue="gray", neutral_hue="slate"), css=css) as demo:
        model_state = gr.State([])  # Store the model as a state variable
        annotations_folder_state = gr.State([])

        with gr.Row():
            # --- SIDEBAR ---
            with gr.Column(scale=1, min_width=300, elem_classes=["glass-card"]):
                gr.Markdown("### Model Configuration")
                model_status = gr.Markdown(
                    value="**Status:** <span style=\"color: #ffbb33;\">Model Not Prompted</span>",
                )

                model_dropdown = gr.Dropdown(
                    choices=MODEL_NAMES,
                    value=MODEL_NAMES[0],
                    label="YOLO E Model"
                )
                
                prompts_table = gr.Dataframe(
                    headers=list(prompts_df_new.columns),
                    value=prompts_df_new,
                    interactive=True,
                    label="Class Names & Descriptions",
                    wrap=True
                )

                with gr.Accordion("Negative Prompts", open=False, visible=False):
                    neg_prompts_table = gr.Dataframe(
                        headers=list(prompts_df_new.columns),
                        value=pd.DataFrame({col: [""] for col in prompts_df_new.columns}),
                        interactive=True,
                        label="Negative Class Descriptions (emb * -1)",
                        wrap=True
                    )
                with gr.Row():
                    set_classes_button = gr.Button("Prompt Model", variant="secondary")
                
                gr.Markdown("---")
                gr.Markdown("### Inference Settings")
                conf_slider = gr.Slider(0.01, 1.0, value=0.01, step=0.005, label="Confidence Threshold")
                batch_slider = gr.Slider(1, 8, value=2, step=1, label="Batch Size")
                
                gr.Markdown("---")
                download_output = gr.File(
                    label="Download Prompted Model (.pt)",
                    value=os.path.join("assets", "_"),
                    interactive=False,
                    height="auto",
                )
 
            # --- MAIN AREA (Workspace) ---
            with gr.Column(scale=3, elem_classes=["glass-card"]):
                gr.Markdown("### Input Sources")
                with gr.Row():
                    with gr.Column():
                        gallery = gr.Gallery(
                            label="Image Gallery", 
                            show_label=False, 
                            height="300px",
                            type="filepath", 
                            columns=4,
                            value=DEFAULT_IMAGES
                        )
                    with gr.Column():
                        webcam_img = gr.Image(
                            label="Webcam Snap",
                            sources=["webcam"],
                            height="300px",
                            type="filepath"
                        )
                
                btn = gr.Button("Run Detection", variant="primary", size="lg")
                
                gr.Markdown("### Predictions")
                output_gallery = gr.Gallery(
                    label="Detection Results",
                    show_label=False,
                    type="numpy",
                    columns=2,
                    height="auto"
                )
                
                with gr.Accordion("Details & Export", open=False):
                    with gr.Row():
                        with gr.Column(scale=2):
                            output_table = gr.Dataframe(label="Detection Summary")
                        with gr.Column(scale=1):
                            with gr.Row():
                                get_annotations_btn = gr.Button("Package Annotations")
                            with gr.Row():
                                download_annotations = gr.JSON(label="Annotations JSON")

        # --- EVENT HANDLERS ---
        
        # Model Preparation
        prompts_table.change(
            fn=sync_negative_prompts_columns,
            inputs=[prompts_table, neg_prompts_table],
            outputs=[neg_prompts_table]
        )

        set_classes_button.click(
            fn=set_classes_and_save_model,
            inputs=[prompts_table, neg_prompts_table, model_dropdown],
            outputs=[model_state, download_output, model_status]
        ).then(
            fn=cleanup_temp_model,
            inputs=download_output,
            outputs=None
        )

        # Image Handling
        webcam_img.change(
            fn=handle_webcam_change_event,
            inputs=[webcam_img, gallery],
            outputs=[gallery]
        )

        # Execution
        btn.click(
            fn=detect_objects_stream,
            inputs=[gallery, batch_slider, model_state, conf_slider],
            outputs=[output_gallery, output_table, annotations_folder_state],
        )
        
        # Export
        get_annotations_btn.click(
            fn=zip_folder,
            inputs=[annotations_folder_state],
            outputs=[download_annotations]
        )

    return demo
