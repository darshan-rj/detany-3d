# pages/1_nuScenes.py

import streamlit as st
import numpy as np
import os
from shared_utils import BOX_TRESHOLD, TEXT_TRESHOLD, device, predict, load_random_nuscenes_image, find_and_predict, find_and_predict_nuscenes_video

# --- Page UI ---
st.header("🖼️ nuScenes Detection")
st.markdown("Use a random image or search for a video sequence in the **nuScenes** dataset.")

# --- Global State Initialization ---
if 'image' not in st.session_state: st.session_state.image = None
if 'current_image_path' not in st.session_state: st.session_state.current_image_path = None
if 'uploaded_file_name' not in st.session_state: st.session_state.uploaded_file_name = None

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 1. Load Image")
    
    if st.button("Load Random nuScenes Image", use_container_width=True, help="Loads a random image from the nuScenes validation dataset."):
        st.session_state.image = load_random_nuscenes_image()
        st.session_state.current_image_path = None
        st.session_state.uploaded_file_name = None

    if st.session_state.image:
        st.image(st.session_state.image, caption="Current nuScenes Image", use_container_width=True)
    else:
        st.info("Load an image to proceed with detection.")

    st.markdown("#### 2. Enter a Text Prompt")
    text_input = st.text_input("Text Prompt", "a car . a person", key="nusc_text_input", help="Separate different objects with ' . '")

    with st.expander("Detection Thresholds"):
        box_threshold = st.slider("Box Threshold (2D)", 0.0, 1.0, BOX_TRESHOLD, 0.01, key="nusc_box_thresh")
        text_threshold = st.slider("Text Threshold (2D)", 0.0, 1.0, TEXT_TRESHOLD, 0.01, key="nusc_text_thresh")
        iou_threshold = st.slider("3D IoU Threshold", 0.0, 1.0, 0.2, 0.01, key="nusc_iou_thresh", help="Filters final 3D bounding boxes based on their predicted IoU score.")

    st.markdown("#### 3. Run Options")
    submit_btn = st.button("Run Detection on Current Image", type="primary", use_container_width=True)
    
    st.markdown("---")
    
    with st.expander("Advanced: Search nuScenes & Generate Video"):
        st.info("Searches the dataset for a scene/image matching your prompt.")
        search_btn = st.button("Find & Predict (Single Image)", key="nusc_search_img", use_container_width=True)
        video_btn = st.button("Find & Predict (Video Sequence)", key="nusc_search_video", use_container_width=True)


with col2:
    st.header("3D Detection Result")
    output_container = st.empty()
    output_container.info("Results will be displayed here.")


# --- BUTTON HANDLERS ---
if submit_btn:
    if st.session_state.image is None: st.error("Please load an image first.")
    elif not text_input: st.error("Please enter a text prompt.")
    else:
        with st.spinner("Running prediction..."):
            result_image, pred_boxes_3d = predict(np.array(st.session_state.image), text_input, box_threshold, text_threshold, iou_threshold, return_boxes=True)
            if result_image is not None:
                output_container.image(result_image, caption="Detection Result", use_container_width=True)

if search_btn:
    if not text_input: st.error("Please enter a text prompt to search.")
    else:
        progress_bar = output_container.progress(0, text="Starting nuScenes search...")
        result_image = find_and_predict(None, text_input, box_threshold, text_threshold, iou_threshold, progress_bar) 
        if result_image is not None:
            progress_bar.progress(100, text="Done!")
            output_container.image(result_image, caption="Detection Result from nuScenes Search", use_container_width=True)
        else:
            progress_bar.empty()

if video_btn:
    if not text_input: st.error("Please enter a text prompt to search.")
    else:
        progress_bar = output_container.progress(0, text="Starting nuScenes video search and generation...")
        video_path = find_and_predict_nuscenes_video(text_input, box_threshold, text_threshold, iou_threshold, progress_bar)
        if video_path:
            progress_bar.progress(100, text="Video generation complete! Playing...")
            mime_type = "video/mp4" if video_path.endswith('.mp4') else "video/avi"
            output_container.video(video_path) 
            st.download_button(label=f"Download nuScenes Inference Video ({'MP4' if video_path.endswith('.mp4') else 'AVI'})", data=open(video_path, 'rb').read(), file_name=os.path.basename(video_path), mime=mime_type)
        else:
            progress_bar.empty()