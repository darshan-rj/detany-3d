# pages/2_NOA.py

import streamlit as st
import numpy as np
import os
from shared_utils import BOX_TRESHOLD, TEXT_TRESHOLD, device, predict, load_random_noa_image, find_and_predict_noa, get_noa_metadata

# --- Page UI ---
st.header("📸 NOA Detection")
st.markdown("Use a random image or search for a video sequence in the **NOA** dataset. Image loading uses **all pinhole images** across all scenes.")

# --- Global State Initialization (Ensuring state keys are reused across pages) ---
if 'image' not in st.session_state: st.session_state.image = None
if 'current_image_path' not in st.session_state: st.session_state.current_image_path = None
if 'uploaded_file_name' not in st.session_state: st.session_state.uploaded_file_name = None

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 1. Load Image")
    
    if st.button("Load Random NOA Image (All Pinhole)", use_container_width=True, help="Loads a random pinhole image from the entire NOA dataset."):
        image, path = load_random_noa_image()
        st.session_state.image = image
        st.session_state.current_image_path = path
        st.session_state.uploaded_file_name = None
        if path and os.path.exists(path):
             st.info(f"Loaded image from: {os.path.basename(path)}")

    if st.session_state.image:
        st.image(st.session_state.image, caption="Current NOA Image", use_container_width=True)
    else:
        st.info("Load an image to proceed with detection.")

    st.markdown("#### 2. Enter a Text Prompt")
    text_input = st.text_input("Text Prompt", "a truck . a bicycle", key="noa_text_input", help="Separate different objects with ' . '")

    with st.expander("Detection Thresholds"):
        box_threshold = st.slider("Box Threshold (2D)", 0.0, 1.0, BOX_TRESHOLD, 0.01, key="noa_box_thresh")
        text_threshold = st.slider("Text Threshold (2D)", 0.0, 1.0, TEXT_TRESHOLD, 0.01, key="noa_text_thresh")
        iou_threshold = st.slider("3D IoU Threshold", 0.0, 1.0, 0.2, 0.01, key="noa_iou_thresh", help="Filters final 3D bounding boxes based on their predicted IoU score.")

    st.markdown("#### 3. Run Options")
    submit_btn = st.button("Run Detection on Current Image", type="primary", use_container_width=True)
    
    st.markdown("---")
    
    with st.expander("Advanced: Search NOA & Generate Video"):
        st.info("Searches a relevant scene in NOA matching your prompt and generates a video.")
        video_btn = st.button("Find & Predict (Video Sequence)", key="noa_search_video", use_container_width=True)


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
                
                # Display Calibration info for NOA image if available
                if st.session_state.current_image_path and './Traces_NOA_structured' in st.session_state.current_image_path:
                    # NOTE: get_noa_metadata must be correctly defined/imported
                    try:
                        _, calib_data = get_noa_metadata(st.session_state.current_image_path)
                        st.sidebar.success(f"Loaded Calibration/Annotations for frame. Example Calib: fx={calib_data['fx']:.0f}, fy={calib_data['fy']:.0f}")
                    except NameError:
                        st.sidebar.warning("Calibration data unavailable (get_noa_metadata not found).")
                    except Exception as e:
                        st.sidebar.warning(f"Error loading calib data: {e}")

if video_btn:
    if not text_input: st.error("Please enter a text prompt to search.")
    else:
        progress_bar = output_container.progress(0, text="Starting NOA video search and generation...")
        video_path = find_and_predict_noa(text_input, box_threshold, text_threshold, iou_threshold, progress_bar)
        if video_path:
            progress_bar.progress(100, text="Video generation complete! Playing...")
            mime_type = "video/mp4" if video_path.endswith('.mp4') else "video/avi"
            output_container.video(video_path) 
            st.download_button(label=f"Download NOA Inference Video ({'MP4' if video_path.endswith('.mp4') else 'AVI'})", data=open(video_path, 'rb').read(), file_name=os.path.basename(video_path), mime=mime_type)
        else:
            progress_bar.empty()