import os
import time
import base64
import cv2
import numpy as np
from PIL import Image, ImageDraw
from dotenv import load_dotenv
import threading  # Import Python's standard threading library

import flet as ft
from flet import (
    AppBar, Page, Container, Text, View, FontWeight,
    TextButton, padding, ThemeMode, border_radius,
    FilePicker, FilePickerResultEvent, icons, MainAxisAlignment,
    CrossAxisAlignment, ScrollMode, ImageFit, colors
)

# --- HUAWEI CLOUD SDK IMPORTS ---
from huaweicloudsdkcore.auth.credentials import BasicCredentials
from huaweicloudsdkcore.exceptions import exceptions
from huaweicloudsdkocr.v1 import *
from huaweicloudsdkocr.v1.region.ocr_region import OcrRegion

# --- CONFIGURATION ---
# Load environment variables from .env file
load_dotenv()

# Huawei Cloud Credentials
AK = os.getenv("HC_OCR_ACCESS_KEY_ID")
SK = os.getenv("HC_OCR_SECRET_ACCESS_KEY_ID")
REGION = os.getenv("HC_OCR_REGION")
PROJECT_ID = os.getenv("HC_OCR_PROJECT_ID")
ENDPOINT = f"ocr.{REGION}.myhuaweicloud.com"

# --- GLOBAL VARIABLES ---
# OpenCV camera setup
cap = cv2.VideoCapture(0)
deb_capt = False # Flag to stop camera capture loop

# --- HUAWEI OCR FUNCTIONS ---

def get_ocr_result_with_boxes(image_path: str):
    """
    Calls the Huawei Cloud General Text OCR API.

    Args:
        image_path: The local path to the image file.

    Returns:
        A tuple containing:
        - A string of all recognized text.
        - A list of location coordinates for bounding boxes.
        - An error message string (or None if successful).
    """
    try:
        # 1. Set up credentials for authentication
        credentials = BasicCredentials(ak=AK, sk=SK, project_id=PROJECT_ID)

        # 2. Initialize the OCR client
        client = OcrClient.new_builder() \
            .with_credentials(credentials) \
            .with_endpoint(ENDPOINT) \
            .build()

        # 3. Read the image file and encode it to Base64
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

        # 4. Construct the request body
        request_body = GeneralTextRequestBody(image=image_base64)
        request = RecognizeGeneralTextRequest(body=request_body)

        # 5. Call the API
        response = client.recognize_general_text(request)

        # 6. Process the JSON response to extract text and locations
        if response.result and response.result.words_block_list:
            all_text = []
            locations = []
            for block in response.result.words_block_list:
                all_text.append(block.words)
                locations.append(block.location)
            return "\n".join(all_text), locations, None
        else:
            return "No text found in the image.", [], None

    except exceptions.ClientRequestException as e:
        error_msg = f"API Error: {e.error_code}\n{e.error_msg}"
        print(f"An error occurred: {e.error_code} - {e.error_msg}")
        return None, None, error_msg
    except FileNotFoundError:
        error_msg = f"Error: The file was not found at {image_path}"
        print(error_msg)
        return None, None, error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred: {e}"
        print(error_msg)
        return None, None, error_msg

def draw_bounding_boxes_huawei(image_path, locations):
    """
    Draws bounding boxes on an image based on coordinates from Huawei OCR.

    Args:
        image_path: The path to the original image.
        locations: A list of coordinate lists for each text block.

    Returns:
        The path to the new image with bounding boxes drawn on it.
    """
    if not locations:
        return image_path

    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    for loc in locations:
        # The 'location' from Huawei API is a list of 4 [x, y] points
        # ImageDraw.polygon can draw this directly
        polygon_points = [tuple(p) for p in loc]
        draw.polygon(polygon_points, outline="red", width=3)

    # Save the annotated image to a new file
    base_name, ext = os.path.splitext(os.path.basename(image_path))
    # Ensure the output directory exists
    if not os.path.exists("output"):
        os.makedirs("output")
    annotated_image_path = f"output/{base_name}_annotated{ext}"
    image.save(annotated_image_path)
    return annotated_image_path


# --- FLET APP MAIN FUNCTION ---

def main(page: Page):
    page.title = "OCR Project"
    page.theme_mode = ThemeMode.LIGHT
    page.window.width = 1200
    page.window.height = 800

    page.fonts = {
        "RobotoFlex": "fonts/RobotoFlex-VariableFont.ttf",
        "RobotoMono": "fonts/RobotoMono-VariableFont.ttf",
        "RobotoMonoItalic": "fonts/RobotoMono-Italic-VariableFont.ttf",
        "Minecraft": "fonts/minecraft_font.ttf"
    }

    # --- UI Components ---

    # For Camera View
    captured_frame_image = ft.Image(
        src="assets/placeholder.jpg",
        width=400,
        height=500,
        fit=ImageFit.CONTAIN
    )

    # For Upload View
    result_image_upload = ft.Image(
        src="assets/result_initial.png",
        width=400,
        height=500,
        border_radius=border_radius.all(10),
        fit=ImageFit.CONTAIN,
        visible=False
    )

    # For Camera View (after processing)
    result_image_cam = ft.Image(
        src="assets/result_initial.png",
        width=400,
        height=500,
        border_radius=border_radius.all(10),
        fit=ImageFit.CONTAIN,
        visible=False
    )

    # --- Camera Logic ---
    def capture_frame_loop():
        """Continuously updates the image control with frames from the camera."""
        cap = cv2.VideoCapture(0)
        try:
            while not deb_capt: # Loop until deb_capt flag is True
                ret, frame = cap.read()
                if ret:
                    _, buffer = cv2.imencode('.png', frame)
                    png_as_text = base64.b64encode(buffer).decode('utf-8')
                    captured_frame_image.src_base64 = png_as_text
                    if captured_frame_image.page: # Check if control is on page
                        captured_frame_image.update()
                time.sleep(0.03) # Small delay to prevent high CPU usage
        finally:
            cap.release()

    def trigger_capture(e):
        """Stops the camera loop and saves the current frame to a file."""
        global deb_capt
        deb_capt = True # Signal the loop to stop
        time.sleep(0.1) # Give the loop a moment to exit

        src_base64_img = captured_frame_image.src_base64
        if src_base64_img:
            img_data = base64.b64decode(src_base64_img)
            # Ensure output directory exists
            if not os.path.exists("output"):
                os.makedirs("output")
            with open("output/capture.png", "wb") as f:
                f.write(img_data)
            print("Frame captured and saved to output/capture.png")

    def clear_capture(e):
        """Resets the camera view to start capturing again."""
        global deb_capt
        deb_capt = False
        captured_frame_image.visible = True
        result_image_cam.visible = False
        prompt_container_cam.visible = False
        capture_button.visible = True
        process_button.visible = True
        retake_button.visible = True
        restart_button_cam.visible = False
        
        if page.route == "/camocr":
            page.update()
            # Restart the camera loop in a new thread
            # FIX: Use standard 'threading' module
            threading.Thread(target=capture_frame_loop, daemon=True).start()

    # --- Processing Logic (Shared by Upload and Camera) ---
    def process_and_display(image_path, result_image_ctrl, prompt_display_ctrl, prompt_container_ctrl, final_controls_to_update):
        """
        Generic function to process an image with OCR and update the UI.
        """
        # Show a loading state
        prompt_display_ctrl.content.controls.clear()
        prompt_display_ctrl.content.controls.append(ft.ProgressRing())
        prompt_container_ctrl.visible = True
        page.update()

        # Call Huawei OCR API
        text_result, locations, error = get_ocr_result_with_boxes(image_path)

        # Clear loading indicator
        prompt_display_ctrl.content.controls.clear()

        if error:
            prompt_display_ctrl.content.controls.append(Text(f"Error: {error}", color=colors.RED))
            result_image_ctrl.src = image_path # Show original image on error
        else:
            # Draw bounding boxes
            annotated_path = draw_bounding_boxes_huawei(image_path, locations)
            result_image_ctrl.src = annotated_path
            
            # Display extracted text
            prompt_display_ctrl.content.controls.append(
                ft.TextField(
                    value=text_result,
                    multiline=True,
                    read_only=True,
                    border=ft.InputBorder.NONE,
                    height=400
                )
            )
        
        result_image_ctrl.visible = True
        for ctrl in final_controls_to_update:
            ctrl.update()

    # --- Event Handlers ---
    def on_file_pick(e: FilePickerResultEvent):
        if e.files:
            image_path = e.files[0].path
            select_image_button.visible = False
            restart_button_upload.visible = True
            process_and_display(
                image_path,
                result_image_upload,
                prompt_display_upload,
                prompt_container_upload,
                [result_image_upload, select_image_button, restart_button_upload, prompt_container_upload]
            )

    def on_process_camera_image(e):
        capture_path = "output/capture.png"
        if os.path.exists(capture_path):
            captured_frame_image.visible = False
            restart_button_cam.visible = True
            capture_button.visible = False
            process_button.visible = False
            retake_button.visible = False

            process_and_display(
                capture_path,
                result_image_cam,
                prompt_display_cam,
                prompt_container_cam,
                [result_image_cam, captured_frame_image, restart_button_cam, prompt_container_cam,
                 capture_button, process_button, retake_button]
            )

    def restart_upload_flow(e):
        result_image_upload.visible = False
        select_image_button.visible = True
        restart_button_upload.visible = False
        prompt_container_upload.visible = False
        page.update()

    def restart_cam_flow(e):
        clear_capture(e)

    # --- UI Definitions ---
    # File picker for upload
    file_picker = FilePicker(on_result=on_file_pick)
    page.overlay.append(file_picker)

    # -- Upload Page Components --
    initial_image_button = ft.Image(src="assets/result_initial.png", width=350, height=350, border_radius=border_radius.all(10))
    select_image_button = TextButton(content=initial_image_button, on_click=lambda _: file_picker.pick_files(allow_multiple=False, allowed_extensions=["png", "jpg", "jpeg", "bmp", "tiff"]))
    restart_button_upload = TextButton(content=Text("Start over", font_family="Minecraft", size=14), visible=False, on_click=restart_upload_flow)
    prompt_display_upload = Container(height=450, content=ft.Column(scroll=ScrollMode.ADAPTIVE))
    prompt_container_upload = Container(content=ft.Column([Text("Extracted Text:", font_family="Minecraft", size=16), prompt_display_upload]), visible=False)

    # -- Camera Page Components --
    capture_button = ft.ElevatedButton("Capture", icon=icons.CAMERA, on_click=trigger_capture)
    process_button = ft.ElevatedButton("Process Card", icon=icons.CHECK, on_click=on_process_camera_image)
    retake_button = ft.ElevatedButton("Retake", icon=icons.REFRESH, on_click=clear_capture)
    restart_button_cam = TextButton(content=Text("Start over", font_family="Minecraft", size=14), visible=False, on_click=restart_cam_flow)
    prompt_display_cam = Container(height=450, content=ft.Column(scroll=ScrollMode.ADAPTIVE))
    prompt_container_cam = Container(content=ft.Column([Text("Extracted Text:", font_family="Minecraft", size=16), prompt_display_cam]), visible=False)

    # --- Page Routing ---
    def route_change(e):
        page.views.clear()
        page.views.append(
            View(
                "/",
                [
                    AppBar(title=Text("OCR", font_family="Minecraft", size=40), bgcolor=colors.SURFACE_VARIANT),
                    Container(
                        content=ft.Column([
                            Text("IMAGE-TO-TEXT APPLICATION", font_family="Minecraft", size=40, weight=FontWeight.W_800),
                            Text("Powered by Huawei Cloud OCR", font_family="Minecraft", size=20, weight=FontWeight.W_500),
                        ]),
                        padding=padding.only(left=100, top=70)
                    ),
                    ft.Row([
                        Container(ft.ElevatedButton("Upload Card", on_click=lambda _: page.go("/ocr")), padding=padding.only(left=100, top=70)),
                        Container(ft.ElevatedButton("Capture Card", on_click=lambda _: page.go("/camocr")), padding=padding.only(left=100, top=70)),
                    ])
                ],
                vertical_alignment=MainAxisAlignment.START,
                horizontal_alignment=CrossAxisAlignment.START
            )
        )
        if page.route == "/camocr":
            global deb_capt
            deb_capt = False # Reset flag when navigating to camera page
            page.views.append(
                View(
                    "/camocr",
                    [
                        AppBar(title=Text("Capture Card"), bgcolor=colors.SURFACE_VARIANT),
                        ft.Row(
                            [
                                ft.Column([
                                    captured_frame_image,
                                    result_image_cam,
                                    ft.Row([capture_button, process_button, retake_button], alignment=MainAxisAlignment.CENTER),
                                    restart_button_cam
                                ], horizontal_alignment=CrossAxisAlignment.CENTER),
                                ft.VerticalDivider(),
                                Container(
                                    content=prompt_container_cam,
                                    width=400,
                                    padding=padding.only(left=10, top=25),
                                    expand=True
                                ),
                            ],
                            vertical_alignment=CrossAxisAlignment.START,
                            expand=True
                        )
                    ],
                )
            )
            # Start the camera capture in a background thread
            # FIX: Use standard 'threading' module
            threading.Thread(target=capture_frame_loop, daemon=True).start()

        if page.route == "/ocr":
            page.views.append(
                View(
                    "/ocr",
                    [
                        AppBar(title=Text("Upload Card"), bgcolor=colors.SURFACE_VARIANT),
                        ft.Row(
                            [
                                ft.Column([
                                    result_image_upload,
                                    select_image_button,
                                    restart_button_upload,
                                ], horizontal_alignment=CrossAxisAlignment.CENTER),
                                ft.VerticalDivider(),
                                Container(
                                    content=prompt_container_upload,
                                    width=400,
                                    padding=padding.only(left=10, top=25),
                                    expand=True
                                ),
                            ],
                            vertical_alignment=CrossAxisAlignment.START,
                            alignment=MainAxisAlignment.CENTER,
                            expand=True
                        )
                    ],
                )
            )
        page.update()

    def view_pop(e):
        page.views.pop()
        top_view = page.views[-1]
        page.go(top_view.route)

    page.on_route_change = route_change
    page.on_view_pop = view_pop
    page.go(page.route)


if __name__ == "__main__":
    if not all([AK, SK, REGION, PROJECT_ID]):
        print("\n" + "="*60)
        print("!!! IMPORTANT: Missing one or more environment variables. !!!")
        print("!!! Please ensure HC_OCR_ACCESS_KEY_ID,                  !!!")
        print("!!! HC_OCR_SECRET_ACCESS_KEY_ID, HC_OCR_REGION, and      !!!")
        print("!!! HC_OCR_PROJECT_ID are set in your .env file.          !!!")
        print("="*60 + "\n")
    else:
        ft.app(target=main, assets_dir="assets")