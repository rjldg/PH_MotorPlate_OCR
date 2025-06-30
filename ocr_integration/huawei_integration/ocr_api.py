import flet as ft
import base64
import os

# Import Huawei Cloud SDK components
from huaweicloudsdkcore.auth.credentials import BasicCredentials
from huaweicloudsdkcore.exceptions import exceptions
from huaweicloudsdkocr.v1 import *
from huaweicloudsdkocr.v1.region.ocr_region import OcrRegion

# --- Configuration ---
# Replace with your actual credentials and project information
# It's recommended to load these from environment variables or a config file for better security
AK = "YOUR_ACCESS_KEY"
SK = "YOUR_SECRET_KEY"
REGION = "ap-southeast-1"  # e.g., 'ap-southeast-1' for CN-Hong Kong
PROJECT_ID = "YOUR_PROJECT_ID"
ENDPOINT = f"ocr.{REGION}.myhuaweicloud.com"

# --- OCR Service Function ---
def get_ocr_result(image_path: str):
    """
    Calls the Huawei Cloud General Text OCR API and returns the extracted text.

    Args:
        image_path: The local path to the image file.

    Returns:
        A string containing all recognized text, or an error message.
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

        # 6. Process the JSON response to extract text
        if response.result and response.result.words_block_list:
            all_text = []
            for block in response.result.words_block_list:
                all_text.append(block.words)
            return "\n".join(all_text)
        else:
            return "No text found in the image."

    except exceptions.ClientRequestException as e:
        # Handle SDK-specific exceptions (e.g., authentication, network)
        print(f"An error occurred: {e.error_code} - {e.error_msg}")
        return f"API Error: {e.error_code}\n{e.error_msg}"
    except FileNotFoundError:
        return f"Error: The file was not found at {image_path}"
    except Exception as e:
        # Handle other potential errors
        print(f"An unexpected error occurred: {e}")
        return f"An unexpected error occurred: {e}"

# --- Flet Application UI and Logic ---
def main(page: ft.Page):
    page.title = "Huawei Cloud OCR - Flet App"
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.theme_mode = ft.ThemeMode.LIGHT
    page.window_width = 800
    page.window_height = 700
    page.scroll = ft.ScrollMode.ADAPTIVE

    # --- UI Components ---

    # Display for the selected image
    selected_image = ft.Image(
        width=400,
        height=300,
        fit=ft.ImageFit.CONTAIN,
        border_radius=ft.border_radius.all(10),
        border=ft.border.all(1, ft.colors.GREY_400)
    )
    
    # Placeholder text when no image is selected
    image_placeholder = ft.Container(
        content=ft.Text("Select an image to start", style=ft.TextThemeStyle.BODY_LARGE, color=ft.colors.GREY_600),
        width=400,
        height=300,
        alignment=ft.alignment.center,
        border_radius=ft.border_radius.all(10),
        border=ft.border.all(1, ft.colors.GREY_400),
    )

    # Text field for OCR results
    ocr_output_text = ft.TextField(
        label="Extracted Text",
        multiline=True,
        read_only=True,
        expand=True,
        border_color=ft.colors.BLUE_GREY_200
    )
    
    # Loading indicator
    progress_ring = ft.ProgressRing(visible=False, width=24, height=24, stroke_width=3)
    
    # Status text
    status_text = ft.Text("")

    def on_dialog_result(e: ft.FilePickerResultEvent):
        """Callback function for when a file is selected."""
        if e.files:
            # Get the path of the first selected file
            image_path = e.files[0].path
            
            # Update the UI
            image_container.content = selected_image # Switch to the image view
            selected_image.src = image_path
            status_text.value = "Image selected. Processing..."
            ocr_output_text.value = "" # Clear previous results
            progress_ring.visible = True
            page.update()

            # Call the OCR function
            result_text = get_ocr_result(image_path)

            # Update UI with the result
            ocr_output_text.value = result_text
            status_text.value = "Processing complete."
            progress_ring.visible = False
            page.update()
        else:
            status_text.value = "File selection was cancelled."
            page.update()

    # File picker dialog
    file_picker = ft.FilePicker(on_result=on_dialog_result)
    page.overlay.append(file_picker)

    # Main image container (shows placeholder or selected image)
    image_container = ft.Container(content=image_placeholder, padding=10)

    # Main layout
    page.add(
        ft.Column(
            [
                ft.Text("General Text OCR", style=ft.TextThemeStyle.HEADLINE_MEDIUM, weight=ft.FontWeight.BOLD),
                ft.Text("Powered by Huawei Cloud", style=ft.TextThemeStyle.BODY_MEDIUM, italic=True),
                ft.Divider(height=20),
                image_container,
                ft.ElevatedButton(
                    "Select Image File",
                    icon=ft.icons.UPLOAD_FILE,
                    on_click=lambda _: file_picker.pick_files(
                        allow_multiple=False,
                        allowed_extensions=["png", "jpg", "jpeg", "bmp", "tiff"]
                    ),
                    style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8))
                ),
                ft.Row([progress_ring, status_text], alignment=ft.MainAxisAlignment.CENTER),
                ft.Container(
                    content=ocr_output_text,
                    padding=10,
                    expand=True,
                )
            ],
            spacing=15,
            alignment=ft.MainAxisAlignment.START,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            expand=True,
        )
    )

# Run the Flet app
if __name__ == "__main__":
    # IMPORTANT: Check if credentials are placeholders
    if "YOUR_ACCESS_KEY" in AK or "YOUR_SECRET_KEY" in SK or "YOUR_PROJECT_ID" in PROJECT_ID:
        print("\n" + "="*60)
        print("!!! IMPORTANT: Please replace the placeholder credentials !!!")
        print("!!! in the Python script (AK, SK, PROJECT_ID) with your   !!!")
        print("!!! actual Huawei Cloud credentials.                      !!!")
        print("="*60 + "\n")
    else:
        ft.app(target=main)