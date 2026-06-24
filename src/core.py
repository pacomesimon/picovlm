import os
import datetime
import tempfile
import torch
import numpy as np
import pandas as pd
import json
from google import genai
from google.genai import types
from PIL import Image
from ultralytics import YOLO
from .utils import batch_iterable

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
MODEL_NAMES = [
                        "yoloe-11s-seg.pt",
                    ]
for model_name in MODEL_NAMES:
    # Construct the absolute path to the model file
    MODEL_PATH = os.path.join(ASSETS_DIR, model_name)

    # Initialize default model globally (or placeholder)
    try:
        # Ensure usage of the moved model file
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Warning: Failed to load model from {MODEL_PATH}. Error: {e}")
        model = None

def set_classes_with_descriptions(class_description_dict, neg_class_description_dict=None, model_name='yoloe-11s-seg.pt'):
    """
    Sets the class names and aggregated prompt embeddings for the model.
    """
    # Load model from assets
    model_path = os.path.join(ASSETS_DIR, model_name)
    local_model = YOLO(model_path)
    
    class_description_dict = dict(class_description_dict)
    if neg_class_description_dict is not None:
        neg_class_description_dict = dict(neg_class_description_dict)
    else:
        neg_class_description_dict = {}
        
    names = tuple(class_description_dict.keys())
    all_pe_s_list = []
    
    for name in names:
        descriptions = tuple(class_description_dict.get(name, []))
        descriptions = [str(i).strip() for i in descriptions 
                        if str(i).strip() != ''
                        ]
        
        neg_descriptions = tuple(neg_class_description_dict.get(name, []))
        neg_descriptions = [str(i).strip() for i in neg_descriptions 
                            if str(i).strip() != ''
                            ]
        pe_list = []
        if descriptions:
            pe_pos = local_model.get_text_pe(descriptions)
            pe_list.append(pe_pos)
            
        if neg_descriptions:
            pe_neg = local_model.get_text_pe(neg_descriptions) * -1.0
            pe_list.append(pe_neg)
            
        if pe_list:
            pe_all = torch.cat(pe_list, dim=1)
            pe_s_aggregated = pe_all.mean(dim=1, keepdim=True)
            all_pe_s_list.append(pe_s_aggregated[0])
        else:
            pe_pos = local_model.get_text_pe([name])
            pe_s_aggregated = pe_pos.mean(dim=1, keepdim=True)
            all_pe_s_list.append(pe_s_aggregated[0])
        
    pe_s_aggregated = torch.cat(all_pe_s_list, dim=0).unsqueeze(0)
    local_model.set_classes(names, pe_s_aggregated)
    return local_model


def detect_objects_stream(images, batch_size=3, model_instance=None, conf_threshold=0.1):
    """
    Run YOLO on images in batches, stream each batch's results.
    """
    # Use global model if none provided, though usually passed from state
    if model_instance is None:
        model_instance = model
        
    if not images:
        yield [], pd.DataFrame([{"Error": "No images uploaded"}]), None
        return
        
    if isinstance(model_instance, list) or model_instance is None:
        yield [], pd.DataFrame([{"Error": "Model is not Prompted"}]), None
        return

    all_counts = {}
    all_results = []
    names = model_instance.names

    # Use temp directory for annotations
    folder_name = tempfile.mkdtemp()
    
    for batch in batch_iterable(images, batch_size):
        # Gradio Gallery with type='filepath' returns list of items.
        # Original code assumed list of tuples/list: [img[0] for img in batch]
        # We preserve this logic.
        batch_paths = [img[0] for img in batch]
        np_batch = [
            np.array(Image.open(img).convert('RGB'))
            for img in batch_paths
            ]

        # Run YOLO prediction
        results = model_instance(np_batch, verbose=False, conf=conf_threshold)

        for res_id, res in enumerate(results):
            # Save annotation to txt
            annotation_filename = os.path.basename(batch_paths[res_id])
            annotation_filename = os.path.splitext(annotation_filename)[0]
            output_path = os.path.join(folder_name, f"{annotation_filename}.txt")
            res.save_txt(output_path)
            
            # Plot
            plotted_rgb = res.plot()
            all_results.append(plotted_rgb)

            # Count classes
            if res.boxes and res.boxes.cls is not None:
                for c in res.boxes.cls.cpu().numpy():
                    class_name = names[int(c)]
                    all_counts[class_name] = all_counts.get(class_name, 0) + 1

        # Create DataFrame
        data = {"number_of_images": [len(all_results)]}
        for class_name in names:
            data[class_name] = [all_counts.get(class_name, 0)]
        df = pd.DataFrame(data)

        # Stream partial result
        yield all_results, df, folder_name


def save_model(model_instance, model_name='yoloe-11s-seg.pt'):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = model_name[:-3] if model_name.endswith('.pt') else model_name
    filename = f"{base_name}__{timestamp}__.pt"
    # Save the model to a temporary directory
    temp_dir = tempfile.gettempdir()
    filepath = os.path.join(temp_dir, filename)
    model_instance.save(filepath)
    return filepath

def set_classes_and_save_model(df, neg_df=None, model_name='yoloe-11s-seg.pt'):
    model_instance = set_classes_with_descriptions(df, neg_df, model_name)
    status = f'**Status:** <span style="color: #00ffcc;">{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Model Prompted successfully!</span>'
    return model_instance, save_model(model_instance, model_name), status

def refine_prompts_with_gemini(prompts_df, output_images, api_key_input=None):
    """
    Refines prompts using Gemini based on previous prompts and detection results.
    """
    # Prioritize user input API key, fallback to environment variable
    api_key = api_key_input or False #os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        status = f'**Status:** <span style="color: #ff4444;">{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Error - No Gemini API Key provided.</span>'
        return prompts_df, status

    client = genai.Client(api_key=api_key)

    # Convert prompts_df to dict
    prompts_dict = prompts_df.to_dict(orient='list')
    cleaned_prompts = {k: [v for v in l if pd.notna(v) and v != ""] for k, l in prompts_dict.items()}

    # Create the function declaration for Gemini
    properties = {
        k: {
            "type": "STRING", 
            "description": f"A better, more descriptive prompt for the class '{k}' to improve detection accuracy."
        } for k in cleaned_prompts.keys()
    }
    
    declaration = {
        "name": "update_prompts",
        "description": "Updates the prompts for target classes based on visual feedback from previous detections.",
        "parameters": {
            "type": "OBJECT",
            "properties": properties,
            "required": list(cleaned_prompts.keys())
        }
    }

    # Prepare visual context
    contents = [
        "I am working on an object detection task. Here are the current prompts/descriptions for each class:",
        json.dumps(cleaned_prompts, indent=2),
        "Based on the provided images (which show the current detection results with bounding boxes), "
        "please refine the prompts to be more specific and helpful for the detection model. "
        "Use the 'update_prompts' function to return the new prompts."
    ]

    if output_images:
        for item in output_images:
            img_data = item
            if isinstance(item, (list, tuple)):
                img_data = item[0]
            
            if isinstance(img_data, np.ndarray):
                img = Image.fromarray(img_data.astype('uint8'))
                contents.append(img)
            elif isinstance(img_data, str) and os.path.exists(img_data):
                img = Image.open(img_data)
                contents.append(img)

    # Call Gemini
    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview", # Using a stable model name compatible with the new SDK
            contents=contents,
            config=types.GenerateContentConfig(
                tools=[types.Tool(
                    function_declarations=[declaration]
                )],
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode="ANY",
                        allowed_function_names=["update_prompts"]
                    )
                )
            )
        )
        
        # Extract the function call
        for part in response.candidates[0].content.parts:
            if part.function_call:
                new_prompts_map = dict(part.function_call.args)
                new_df_data = {k: [new_prompts_map.get(k, "")] for k in prompts_df.columns}
                new_df = pd.DataFrame(new_df_data)
                status = f'**Status:** <span style="color: #00ffcc;">{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Prompts Refined via AI</span>'
                return new_df, status
                
    except Exception as e:
        status = f'**Status:** <span style="color: #ff4444;">{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Error during Gemini refinement: {str(e)}</span>'
        return prompts_df, status

    status = f'**Status:** <span style="color: #ffbb33;">{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Gemini completed but no changes were made.</span>'
    return prompts_df, status
