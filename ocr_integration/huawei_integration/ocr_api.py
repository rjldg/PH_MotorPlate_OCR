import flet as ft
import base64
import os

from dotenv import load_dotenv

# Import Huawei Cloud SDK components
from huaweicloudsdkcore.auth.credentials import BasicCredentials
from huaweicloudsdkcore.exceptions import exceptions
from huaweicloudsdkocr.v1 import *
from huaweicloudsdkocr.v1.region.ocr_region import OcrRegion

# --- Configuration ---
# Loads credentials from a .env file in the same directory
load_dotenv()
AK = os.getenv("HC_OCR_ACCESS_KEY_ID")
SK = os.getenv("HC_OCR_SECRET_ACCESS_KEY_ID")
REGION = os.getenv("HC_OCR_REGION")
PROJECT_ID = os.getenv("HC_OCR_PROJECT_ID")
ENDPOINT = f"ocr.{REGION}.myhuaweicloud.com"

# --- OCR Service Function ---
def get_ocr_result(image_path: str):
    """
    Calls the Huawei Cloud General Text OCR API and returns the extracted text.
    """
    try:
        credentials = BasicCredentials(ak=AK, sk=SK, project_id=PROJECT_ID)
        client = OcrClient.new_builder() \
            .with_credentials(credentials) \
            .with_endpoint(ENDPOINT) \
            .build()

        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

        request_body = GeneralTextRequestBody(image=image_base64)
        request = RecognizeGeneralTextRequest(body=request_body)

        response = client.recognize_general_text(request)

        if response.result and response.result.words_block_list:
            all_text = [block.words for block in response.result.words_block_list]
            return "\n".join(all_text)
        else:
            return "No text found in the image."

    except exceptions.ClientRequestException as e:
        print(f"An error occurred: {e.error_code} - {e.error_msg}")
        return f"API Error: {e.error_code}\n{e.error_msg}"
    except FileNotFoundError:
        return f"Error: The file was not found at {image_path}"
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return f"An unexpected error occurred: {e}"

# --- Flet Application UI and Logic ---
def main(page: ft.Page):
    page.title = "Huawei Cloud OCR - Flet App"
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.theme_mode = ft.ThemeMode.LIGHT
    
    # DEPRECATION FIX: Use page.window.width and page.window.height
    page.window.width = 800
    page.window.height = 700
    page.scroll = ft.ScrollMode.ADAPTIVE

    # --- UI Components ---

    # FIX: Removed 'border' and 'border_radius' from ft.Image
    selected_image = ft.Image(
        width=400,
        height=300,
        fit=ft.ImageFit.CONTAIN,
    )
    
    # This container holds the placeholder text initially
    image_placeholder = ft.Container(
        content=ft.Text("Select an image to start", style=ft.TextThemeStyle.BODY_LARGE, color=ft.colors.GREY_600),
        width=400,
        height=300,
        alignment=ft.alignment.center,
        border_radius=ft.border_radius.all(10),
        border=ft.border.all(1, ft.colors.GREY_400),
    )

    ocr_output_text = ft.TextField(
        label="Extracted Text",
        multiline=True,
        read_only=True,
        expand=True,
        border_color=ft.colors.BLUE_GREY_200
    )
    
    progress_ring = ft.ProgressRing(visible=False, width=24, height=24, stroke_width=3)
    status_text = ft.Text("")

    def on_dialog_result(e: ft.FilePickerResultEvent):
        if e.files:
            image_path = e.files[0].path
            
            image_container.content = selected_image 
            selected_image.src = image_path
            status_text.value = "Image selected. Processing..."
            ocr_output_text.value = "" 
            progress_ring.visible = True
            page.update()

            result_text = get_ocr_result(image_path)

            ocr_output_text.value = result_text
            status_text.value = "Processing complete."
            progress_ring.visible = False
            page.update()
        else:
            status_text.value = "File selection was cancelled."
            page.update()

    file_picker = ft.FilePicker(on_result=on_dialog_result)
    page.overlay.append(file_picker)

    # FIX: Apply the border and radius to the container that holds the image.
    image_container = ft.Container(
        content=image_placeholder,
        padding=10,
        border_radius=ft.border_radius.all(10),
        border=ft.border.all(1, ft.colors.GREY_400)
    )

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

if __name__ == "__main__":
    # FIX: More robust check for credentials
    if not all([AK, SK, REGION, PROJECT_ID]):
        print("\n" + "="*60)
        print("!!! IMPORTANT: Missing one or more environment variables. !!!")
        print("!!! Please ensure HC_OCR_ACCESS_KEY_ID,                  !!!")
        print("!!! HC_OCR_SECRET_ACCESS_KEY_ID, HC_OCR_REGION, and      !!!")
        print("!!! HC_OCR_PROJECT_ID are set in your .env file.          !!!")
        print("="*60 + "\n")

        print(SK)
        print(AK)
        print(PROJECT_ID)
        print(REGION)
    else:
        ft.app(target=main)