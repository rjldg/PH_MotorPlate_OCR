from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

class MotorcycleDAL:
    """
    A Data Access Layer for handling CRUD operations for motorcycle documents
    in a MongoDB collection. It ensures a unique index on 'plate_number'.
    """

    def __init__(self, uri: str, db_name: str = 'motorcycle_db', collection_name: str = 'motorcycles'):
        """
        Initializes the connection to the MongoDB database and ensures the
        unique index on 'plate_number' exists.

        Args:
            uri (str): The MongoDB connection string.
            db_name (str): The name of the database.
            collection_name (str): The name of the collection.
        """
        try:
            self.client = MongoClient(uri)
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
            self._create_unique_index()
            print("Successfully connected to the database.")
        except Exception as e:
            print(f"Error: Could not connect to the database. {e}")
            self.client = None
            self.collection = None

    def _create_unique_index(self):
        """
        (Private) Creates a unique index on the 'plate_number' field to prevent duplicates.
        This method is idempotent; it won't re-create the index if it already exists.
        """
        if self.collection is not None:
            try:
                self.collection.create_index("plate_number", unique=True)
                print("Unique index on 'plate_number' is ensured.")
            except Exception as e:
                print(f"Error creating unique index: {e}")

    def insert_motorcycle(self, plate_number: str, region: str, blacklisted: bool = False, expired: bool = False, violations: bool = False) -> bool:
        """
        1. Inserts a new motorcycle document into the collection.

        Args:
            plate_number (str): The unique plate number of the motorcycle.
            region (str): The region where the motorcycle is registered.
            blacklisted (bool): Status if the motorcycle is blacklisted.
            expired (bool): Status if the registration is expired.
            violations (bool): Status if the motorcycle has violations.

        Returns:
            bool: True if insertion was successful, False otherwise.
        """
        if self.collection is None:
            return False
        try:
            motorcycle_doc = {
                "plate_number": plate_number,
                "region": region,
                "blacklisted": blacklisted,
                "expired": expired,
                "violations": violations
            }
            self.collection.insert_one(motorcycle_doc)
            print(f"Successfully inserted motorcycle with plate: {plate_number} in region: {region}")
            return True
        except DuplicateKeyError:
            print(f"Error: A motorcycle with plate number '{plate_number}' already exists.")
            return False
        except Exception as e:
            print(f"An unexpected error occurred during insertion: {e}")
            return False

    def _update_status_flag(self, plate_number: str, field_to_update: str, status: bool) -> bool:
        """
        (Private) Generic helper to update a single boolean status field for a motorcycle.
        """
        if self.collection is None:
            return False
        try:
            result = self.collection.update_one(
                {"plate_number": plate_number},
                {"$set": {field_to_update: status}}
            )
            if result.matched_count > 0:
                print(f"Successfully updated '{field_to_update}' for plate {plate_number} to {status}.")
                return True
            else:
                print(f"Error: No motorcycle found with plate number '{plate_number}'.")
                return False
        except Exception as e:
            print(f"An unexpected error occurred during update: {e}")
            return False

    def update_blacklisted_status(self, plate_number: str) -> bool:
        """2. Sets the 'blacklisted' status to True for a given plate number."""
        return self._update_status_flag(plate_number, "blacklisted", True)

    def update_expired_status(self, plate_number: str) -> bool:
        """3. Sets the 'expired' status to True for a given plate number."""
        return self._update_status_flag(plate_number, "expired", True)

    def update_violations_status(self, plate_number: str) -> bool:
        """4. Sets the 'violations' status to True for a given plate number."""
        return self._update_status_flag(plate_number, "violations", True)

    def clear_all_statuses(self, plate_number: str) -> bool:
        """
        5. Clears all statuses (blacklisted, expired, violations) to False
           for a given plate number.
        """
        if self.collection is None:
            return False
        try:
            statuses_to_clear = {
                "blacklisted": False,
                "expired": False,
                "violations": False
            }
            result = self.collection.update_one(
                {"plate_number": plate_number},
                {"$set": statuses_to_clear}
            )
            if result.matched_count > 0:
                print(f"Successfully cleared all statuses for plate: {plate_number}.")
                return True
            else:
                print(f"Error: No motorcycle found with plate number '{plate_number}'.")
                return False
        except Exception as e:
            print(f"An unexpected error occurred while clearing statuses: {e}")
            return False

    def delete_motorcycle(self, plate_number: str) -> bool:
        """6. Deletes a motorcycle document from the collection by its plate number."""
        if self.collection is None:
            return False
        try:
            result = self.collection.delete_one({"plate_number": plate_number})
            if result.deleted_count > 0:
                print(f"Successfully deleted motorcycle with plate: {plate_number}")
                return True
            else:
                print(f"Error: No motorcycle found with plate number '{plate_number}' to delete.")
                return False
        except Exception as e:
            print(f"An unexpected error occurred during deletion: {e}")
            return False

    def find_motorcycle(self, plate_number: str):
        """Finds and returns a single motorcycle document by its plate number."""
        if self.collection is None:
            return None
        return self.collection.find_one({"plate_number": plate_number})

    def close_connection(self):
        """Closes the connection to the database."""
        if self.client:
            self.client.close()
            print("Database connection closed.")


# --- FLET APP IMPORTS AND SETUP ---
import os
import time
import base64
import cv2
import numpy as np
from PIL import Image, ImageDraw
from dotenv import load_dotenv
import threading

import flet as ft
from flet import (
    AppBar, Page, Container, Text, View, FontWeight,
    TextButton, padding, ThemeMode, border_radius,
    FilePicker, FilePickerResultEvent, icons, MainAxisAlignment,
    CrossAxisAlignment, ScrollMode, ImageFit, colors, Row, Column,
    ElevatedButton, TextField, ProgressRing, VerticalDivider, InputBorder
)

# --- HUAWEI CLOUD SDK IMPORTS ---
from huaweicloudsdkcore.auth.credentials import BasicCredentials
from huaweicloudsdkcore.exceptions import exceptions
from huaweicloudsdkocr.v1 import *
from huaweicloudsdkocr.v1.region.ocr_region import OcrRegion

# --- CONFIGURATION ---
load_dotenv()

# Huawei Cloud Credentials
AK = os.getenv("HC_OCR_ACCESS_KEY_ID")
SK = os.getenv("HC_OCR_SECRET_ACCESS_KEY_ID")
REGION = os.getenv("HC_OCR_REGION")
PROJECT_ID = os.getenv("HC_OCR_PROJECT_ID")
ENDPOINT = f"ocr.{REGION}.myhuaweicloud.com"

# MongoDB Credentials
MONGO_URI = os.getenv("MONGO_URI")

# --- GLOBAL VARIABLES ---
cap = cv2.VideoCapture(0)
deb_capt = False

# --- HUAWEI OCR FUNCTIONS ---
def get_ocr_result_with_boxes(image_path: str):
    """
    Calls Huawei OCR and distinguishes plate number from region based on vertical position.
    Returns: plate_number, region, all_locations, error_message
    """
    try:
        credentials = BasicCredentials(ak=AK, sk=SK, project_id=PROJECT_ID)
        client = OcrClient.new_builder().with_credentials(credentials).with_endpoint(ENDPOINT).build()
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")
        
        request_body = GeneralTextRequestBody(image=image_base64)
        request = RecognizeGeneralTextRequest(body=request_body)
        response = client.recognize_general_text(request)

        if response.result and response.result.words_block_list:
            # Sort blocks by their vertical position (top to bottom)
            # The location is a list of 4 [x, y] points. We use the y-coord of the first point.
            sorted_blocks = sorted(response.result.words_block_list, key=lambda b: b.location[0][1])
            
            all_locations = [block.location for block in sorted_blocks]
            
            plate_number = ""
            region = ""

            if len(sorted_blocks) >= 1:
                plate_number = sorted_blocks[0].words.strip()
            if len(sorted_blocks) >= 2:
                region = sorted_blocks[1].words.strip()

            return plate_number, region, all_locations, None
        else:
            return "No text found.", "", [], None
            
    except exceptions.ClientRequestException as e:
        return None, None, None, f"API Error: {e.error_code}\n{e.error_msg}"
    except Exception as e:
        return None, None, None, f"An unexpected error occurred: {e}"

def draw_bounding_boxes_huawei(image_path, locations):
    if not locations:
        return image_path
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    for loc in locations:
        polygon_points = [tuple(p) for p in loc]
        draw.polygon(polygon_points, outline="red", width=3)
    if not os.path.exists("output"):
        os.makedirs("output")
    base_name, ext = os.path.splitext(os.path.basename(image_path))
    annotated_image_path = f"output/{base_name}_annotated{ext}"
    image.save(annotated_image_path)
    return annotated_image_path

# --- FLET APP MAIN FUNCTION ---
def main(page: Page):
    page.title = "License Plate OCR and DB Management"
    page.theme_mode = ThemeMode.DARK
    page.window.width = 1200
    page.window.height = 650

    # --- DAL INSTANCE ---
    dal = MotorcycleDAL(MONGO_URI)

    # --- UI Components ---
    captured_frame_image = ft.Image(src="assets/placeholder.jpg", width=400, height=300, fit=ImageFit.CONTAIN)
    result_image_upload = ft.Image(src="assets/result_initial.png", width=400, height=300, border_radius=border_radius.all(10), fit=ImageFit.CONTAIN, visible=False)
    result_image_cam = ft.Image(src="assets/result_initial.png", width=400, height=300, border_radius=border_radius.all(10), fit=ImageFit.CONTAIN, visible=False)
    
    # Text fields for detected plate and region
    detected_plate_field = TextField(label="Detected Plate Number", read_only=True, width=400)
    detected_region_field = TextField(label="Detected Region", read_only=True, width=400)
    
    # A dedicated container for progress/error messages
    status_message_container = Container(content=None, height=50)

    # --- Database Interaction Buttons and Indicators ---
    add_button = ElevatedButton("Add to DB", icon=icons.ADD, disabled=True)
    add_status = Text("", size=12, color=colors.ORANGE_700)
    
    blacklist_button = ElevatedButton("Flag as Blacklisted", icon=icons.BLOCK, disabled=True)
    blacklist_status = Text("", size=12, color=colors.ORANGE_700)

    expire_button = ElevatedButton("Flag as Expired", icon=icons.TIMER_OFF, disabled=True)
    expire_status = Text("", size=12, color=colors.ORANGE_700)

    violation_button = ElevatedButton("Flag Violation History", icon=icons.GAVEL, disabled=True)
    violation_status = Text("", size=12, color=colors.ORANGE_700)
    
    delete_button = ElevatedButton("Delete License", icon=icons.DELETE, disabled=True, bgcolor=colors.RED_200, color=colors.BLACK)
    delete_status = Text("", size=12, color=colors.ORANGE_700)

    db_controls_container = Column([
        Row([add_button, add_status], alignment=MainAxisAlignment.START),
        Row([blacklist_button, blacklist_status], alignment=MainAxisAlignment.START),
        Row([expire_button, expire_status], alignment=MainAxisAlignment.START),
        Row([violation_button, violation_status], alignment=MainAxisAlignment.START),
        Row([delete_button, delete_status], alignment=MainAxisAlignment.START),
    ], spacing=10, visible=False)

    # --- DB Button Logic ---
    def update_db_buttons_state(plate_number):
        if not plate_number:
            db_controls_container.visible = False
            page.update()
            return

        record = dal.find_motorcycle(plate_number)
        db_controls_container.visible = True

        if record:
            # Plate exists in DB
            add_button.disabled = True
            add_status.value = "Plate already exists in DB."
            delete_button.disabled = False
            delete_status.value = ""

            # Check blacklisted status
            blacklist_button.disabled = record.get('blacklisted', False)
            blacklist_status.value = "Already blacklisted." if blacklist_button.disabled else ""
            
            # Check expired status
            expire_button.disabled = record.get('expired', False)
            expire_status.value = "Already expired." if expire_button.disabled else ""

            # Check violations status
            violation_button.disabled = record.get('violations', False)
            violation_status.value = "Already has violations." if violation_button.disabled else ""
        else:
            # Plate does not exist in DB
            add_button.disabled = False
            add_status.value = ""
            delete_button.disabled = True
            delete_status.value = "Plate not in DB."
            blacklist_button.disabled = True
            blacklist_status.value = "Plate not in DB."
            expire_button.disabled = True
            expire_status.value = "Plate not in DB."
            violation_button.disabled = True
            violation_status.value = "Plate not in DB."
        
        page.update()

    # --- DB Button Event Handlers ---
    def add_license(e):
        plate = detected_plate_field.value
        region = detected_region_field.value or "REGION UNKNOWN"
        if plate:
            dal.insert_motorcycle(plate, region)
            update_db_buttons_state(plate)

    def flag_blacklisted(e):
        plate = detected_plate_field.value
        if plate:
            dal.update_blacklisted_status(plate)
            update_db_buttons_state(plate)

    def flag_expired(e):
        plate = detected_plate_field.value
        if plate:
            dal.update_expired_status(plate)
            update_db_buttons_state(plate)

    def flag_violations(e):
        plate = detected_plate_field.value
        if plate:
            dal.update_violations_status(plate)
            update_db_buttons_state(plate)
            
    def delete_license(e):
        plate = detected_plate_field.value
        if plate:
            dal.delete_motorcycle(plate)
            update_db_buttons_state(plate)

    add_button.on_click = add_license
    blacklist_button.on_click = flag_blacklisted
    expire_button.on_click = flag_expired
    violation_button.on_click = flag_violations
    delete_button.on_click = delete_license

    # --- Camera Logic ---
    def capture_frame_loop():
        cap = cv2.VideoCapture(0)
        try:
            while not deb_capt:
                ret, frame = cap.read()
                if ret:
                    _, buffer = cv2.imencode('.png', frame)
                    png_as_text = base64.b64encode(buffer).decode('utf-8')
                    captured_frame_image.src_base64 = png_as_text
                    if captured_frame_image.page:
                        captured_frame_image.update()
                time.sleep(0.03)
        finally:
            cap.release()

    def trigger_capture(e):
        global deb_capt
        deb_capt = True
        time.sleep(0.1)
        src_base64_img = captured_frame_image.src_base64
        if src_base64_img:
            img_data = base64.b64decode(src_base64_img)
            if not os.path.exists("output"):
                os.makedirs("output")
            with open("output/capture.png", "wb") as f:
                f.write(img_data)
            print("Frame captured.")

    def clear_capture(e):
        global deb_capt
        deb_capt = False
        captured_frame_image.visible = True
        result_image_cam.visible = False
        prompt_container_cam.visible = False
        db_controls_container.visible = False
        detected_plate_field.value = ""
        detected_region_field.value = ""
        capture_button.visible = True
        process_button.visible = True
        retake_button.visible = True
        restart_button_cam.visible = False
        if page.route == "/camocr":
            page.update()
            threading.Thread(target=capture_frame_loop, daemon=True).start()

    # --- Processing Logic ---
    def process_and_display(image_path, result_image_ctrl, prompt_container_ctrl):
        # Show a loading state in the dedicated container
        status_message_container.content = ProgressRing()
        prompt_container_ctrl.visible = True
        page.update()

        plate_number, region, locations, error = get_ocr_result_with_boxes(image_path)

        # Clear loading indicator
        status_message_container.content = None

        if error:
            status_message_container.content = Text(f"Error: {error}", color=colors.RED, size=12)
            result_image_ctrl.src = image_path
            detected_plate_field.value = ""
            detected_region_field.value = ""
            db_controls_container.visible = False
        else:
            annotated_path = draw_bounding_boxes_huawei(image_path, locations)
            result_image_ctrl.src = annotated_path
            detected_plate_field.value = plate_number
            detected_region_field.value = region
            update_db_buttons_state(plate_number)
        
        result_image_ctrl.visible = True
        page.update()

    # --- Event Handlers ---
    def on_file_pick(e: FilePickerResultEvent):
        if e.files:
            image_path = e.files[0].path
            select_image_button.visible = False
            restart_button_upload.visible = True
            process_and_display(image_path, result_image_upload, prompt_container_upload)

    def on_process_camera_image(e):
        capture_path = "output/capture.png"
        if os.path.exists(capture_path):
            captured_frame_image.visible = False
            restart_button_cam.visible = True
            capture_button.visible = False
            process_button.visible = False
            retake_button.visible = False
            process_and_display(capture_path, result_image_cam, prompt_container_cam)
    
    def restart_upload_flow(e):
        result_image_upload.visible = False
        select_image_button.visible = True
        restart_button_upload.visible = False
        prompt_container_upload.visible = False
        db_controls_container.visible = False
        detected_plate_field.value = ""
        detected_region_field.value = ""
        page.update()

    # --- UI Definitions ---
    file_picker = FilePicker(on_result=on_file_pick)
    page.overlay.append(file_picker)
    
    initial_image_button = ft.Image(src="assets/result_initial.png", width=350, height=350, border_radius=border_radius.all(10))
    select_image_button = TextButton(content=initial_image_button, on_click=lambda _: file_picker.pick_files(allow_multiple=False, allowed_extensions=["png", "jpg", "jpeg"]))
    restart_button_upload = TextButton(content=Text("Start over", size=14), visible=False, on_click=restart_upload_flow)
    prompt_container_upload = Container(content=Column([
        Text("OCR Result:", size=16), 
        detected_plate_field,
        detected_region_field,
        status_message_container,
        db_controls_container
    ]), visible=False)

    capture_button = ElevatedButton("Capture", icon=icons.CAMERA, on_click=trigger_capture)
    process_button = ElevatedButton("Process", icon=icons.CHECK, on_click=on_process_camera_image)
    retake_button = ElevatedButton("Retake", icon=icons.REFRESH, on_click=clear_capture)
    restart_button_cam = TextButton(content=Text("Start over", size=14), visible=False, on_click=clear_capture)
    prompt_container_cam = Container(content=Column([
        Text("OCR Result:", size=16), 
        detected_plate_field,
        detected_region_field,
        status_message_container,
        db_controls_container
    ]), visible=False)

    # --- Page Routing ---
    def route_change(e):
        page.views.clear()
        page.views.append(
            View(
                "/",
                [
                    AppBar(title=Text("License Plate OCR"), bgcolor=colors.SURFACE_VARIANT),
                    Container(
                        content=Column([
                            Text("License Plate Recognition System", size=30, weight=FontWeight.W_800),
                            Text("Powered by Huawei Cloud OCR & MongoDB", size=18, weight=FontWeight.W_500),
                        ]),
                        padding=padding.only(left=100, top=70)
                    ),
                    Row([
                        Container(ElevatedButton("Upload Plate Image", on_click=lambda _: page.go("/ocr")), padding=padding.only(left=100, top=50)),
                        Container(ElevatedButton("Capture with Camera", on_click=lambda _: page.go("/camocr")), padding=padding.only(left=50, top=50)),
                    ])
                ],
                vertical_alignment=MainAxisAlignment.START,
                horizontal_alignment=CrossAxisAlignment.START
            )
        )
        if page.route == "/camocr":
            global deb_capt
            deb_capt = False
            page.views.append(
                View(
                    "/camocr",
                    [
                        AppBar(title=Text("Capture Plate"), bgcolor=colors.SURFACE_VARIANT),
                        Row(
                            [
                                Column([
                                    captured_frame_image,
                                    result_image_cam,
                                    Row([capture_button, process_button, retake_button], alignment=MainAxisAlignment.CENTER),
                                    restart_button_cam
                                ], horizontal_alignment=CrossAxisAlignment.CENTER),
                                VerticalDivider(),
                                Container(
                                    content=prompt_container_cam,
                                    width=450,
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
            threading.Thread(target=capture_frame_loop, daemon=True).start()

        if page.route == "/ocr":
            page.views.append(
                View(
                    "/ocr",
                    [
                        AppBar(title=Text("Upload Plate"), bgcolor=colors.SURFACE_VARIANT),
                        Row(
                            [
                                Column([
                                    result_image_upload,
                                    select_image_button,
                                    restart_button_upload,
                                ], horizontal_alignment=CrossAxisAlignment.CENTER),
                                VerticalDivider(),
                                Container(
                                    content=prompt_container_upload,
                                    width=450,
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

    def on_disconnect(e):
        dal.close_connection()
        global deb_capt
        deb_capt = True

    page.on_disconnect = on_disconnect


if __name__ == "__main__":
    if not all([AK, SK, REGION, PROJECT_ID, MONGO_URI]):
        print("\n" + "="*60)
        print("!!! IMPORTANT: Missing one or more environment variables. !!!")
        print("!!! Please ensure HC_OCR_*, and MONGO_URI are set.      !!!")
        print("="*60 + "\n")
    else:
        ft.app(target=main, assets_dir="assets")
