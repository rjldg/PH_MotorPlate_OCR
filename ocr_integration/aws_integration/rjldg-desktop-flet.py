import os
import boto3
import numpy as np
import csv
import mindspore as ms
import time

from dotenv import load_dotenv
from mindspore import Tensor
from mindspore.train import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn import Softmax
from PIL import Image, ImageDraw
import flet as ft
from flet import AppBar, CupertinoFilledButton, Page, Container, Text, View, FontWeight, colors, TextButton, padding, ThemeMode, border_radius, Image as FletImage, FilePicker, FilePickerResultEvent, icons, Column, Row

#from image_classifier.resnet50_archi import resnet50
#from main import predict, retrieve_and_generate_response, ref_class_names

first_prompt_entered = True

# Load environment variables
load_dotenv()
AWS_REGION = os.getenv("AWS_REGION_TEXTRACT")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID_TEXTRACT")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY_TEXTRACT")

# AWS Textract Client
client = boto3.client(
    'textract',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

def ShowBoundingBox(draw, box, width, height, boxColor):
    left = width * box['Left']
    top = height * box['Top'] 
    draw.rectangle([left, top, left + (width * box['Width']), top + (height * box['Height'])], outline=boxColor, width=2)

def ShowSelectedElement(draw, box, width, height, boxColor):
    left = width * box['Left']
    top = height * box['Top'] 
    draw.rectangle([left, top, left + (width * box['Width']), top + (height * box['Height'])], fill=boxColor)

def draw_bounding_boxes(image_path, blocks):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    for block in blocks:
        if 'Geometry' in block and 'BoundingBox' in block['Geometry']:
            box = block['Geometry']['BoundingBox']
            if block['BlockType'] == "KEY_VALUE_SET":
                color = 'red' if block.get('EntityTypes', [''])[0] == "KEY" else 'green'
            elif block['BlockType'] == 'TABLE':
                color = 'blue'
            elif block['BlockType'] == 'CELL':
                color = 'yellow'
            elif block['BlockType'] == 'SELECTION_ELEMENT' and block.get('SelectionStatus') == 'SELECTED':
                ShowSelectedElement(draw, box, width, height, 'blue')
                continue
            else:
                color = 'red'
            ShowBoundingBox(draw, box, width, height, color)

    base_name, ext = os.path.splitext(image_path)
    annotated_image_path = f"{base_name}_annotated{ext}"
    image.save(annotated_image_path)
    return annotated_image_path

def query_document(image_bytes, question):
    response = client.analyze_document(
        Document={'Bytes': image_bytes},
        FeatureTypes=["TABLES", "FORMS", "QUERIES"],
        QueriesConfig={'Queries': [{'Text': question}]}
    )
    
    answer = "No answer found."
    for block in response.get('Blocks', []):
        if block["BlockType"] == "QUERY_RESULT":
            answer = block["Text"]
            break
    
    return answer

def save_to_csv(key_values):
    identity_number = key_values.get('IdentityNumber')
    if not identity_number:
        return "No Identity Number Found"
    filename = f"{identity_number}.csv"
    if os.path.exists(filename):
        return f"Data for ID {identity_number} already exists."
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Key", "Value"])
        for key, value in key_values.items():
            writer.writerow([key, value])
    return f"Data saved as {filename}"

def extract_key_values(response):
    key_values = {}
    for block in response.get('Blocks', []):
        if block['BlockType'] == 'KEY_VALUE_SET' and 'EntityTypes' in block and 'KEY' in block['EntityTypes']:
            key = None
            value = None
            if 'Relationships' in block:
                for rel in block['Relationships']:
                    if rel['Type'] == 'CHILD':
                        key = ''.join([w['Text'] for w in response['Blocks'] if w['Id'] in rel['Ids'] and 'Text' in w])
                    elif rel['Type'] == 'VALUE':
                        for value_id in rel['Ids']:
                            value_block = next((b for b in response['Blocks'] if b['Id'] == value_id), None)
                            if value_block and 'Relationships' in value_block:
                                for v_rel in value_block['Relationships']:
                                    if v_rel['Type'] == 'CHILD':
                                        value = ''.join([w['Text'] for w in response['Blocks'] if w['Id'] in v_rel['Ids'] and 'Text' in w])
            if key and value:
                key_values[key] = value
    return key_values

def main(page: Page):

    page.title = "OCR Project"
    page.theme_mode = ThemeMode.LIGHT

    page.fonts = {
        "RobotoFlex": f"fonts/RobotoFlex-VariableFont.ttf",
        "RobotoMono": f"fonts/RobotoMono-VariableFont.ttf",
        "RobotoMonoItalic": f"fonts/RobotoMono-Italic-VariableFont.ttf",
        "Minecraft": f"fonts/minecraft_font.ttf"
    }
    
    def process_image(e, result_image, file_picker):
        if file_picker.result and file_picker.result.files:
            image_path = file_picker.result.files[0].path

            if os.path.exists(image_path):
                with open(image_path, "rb") as image_file:
                    image_bytes = image_file.read()
                response = client.analyze_document(Document={'Bytes': image_bytes}, FeatureTypes=["TABLES", "FORMS"])
                # For bounding boxes
                blocks = response.get('Blocks', [])
                boxed_image_path = draw_bounding_boxes(image_path, blocks)
                result_image.src = boxed_image_path
                result_image.visible = True

                # For displaying key-value pairs
                key_values = extract_key_values(response)
                message = save_to_csv(key_values)
                
                prompt_container.visible = True
                select_image.visible = False
                restart_button.visible = True

                global first_prompt_entered

                #question = prompt_input.value

                #input_text = query_document(image_bytes, question)

                prompt_text.value = message + '\n'
                key_val = "\n\n".join([f"{key}: {value}" for key, value in key_values.items()])

                type_speed = 0.001

                if first_prompt_entered:
                    prompt_display.content.controls.append(Text("", size=15, font_family="RobotoFlex", weight=FontWeight.W_400, color="#000000"))
                    prompt_display.update()
                    first_prompt_entered = False
                else:
                    prompt_display.content.controls[-1].value = ""
                    prompt_display.update()

                for char in key_val:
                    prompt_display.content.controls[-1].value += char
                    prompt_display.update()
                    time.sleep(type_speed)

                #prompt_input.on_submit = process_query

                result_image.update()
                #prompt_input.update()
                prompt_container.update()
                restart_button.update()
                select_image.update()

    def restart_process(e):
        global first_prompt_entered

        result_image.visible = False
        select_image.visible = True
        restart_button.visible = False
        if not(first_prompt_entered):
            prompt_display.content.controls[-1].value = ""
        #prompt_input.value = ""
        prompt_container.visible = False

        restart_button.update()
        result_image.update()
        select_image.update()
        #prompt_input.update()
        prompt_display.update()
        prompt_container.update()

    restart_button = TextButton(content=Text("Start over", font_family="Minecraft", size=14, weight=FontWeight.W_300, color="#000000"), visible=False, on_click=restart_process)

    initial_image = ft.Image(src="assets/result_initial.png", width=350, height=350, border_radius=border_radius.all(10))
    result_image = ft.Image(src="assets/result_initial.png", width=400, height=400, border_radius=border_radius.all(10), fit=ft.ImageFit.CONTAIN)
    result_image.visible = False

    file_picker = FilePicker()
    file_picker.on_result = lambda e: process_image(e, result_image, file_picker)
    page.overlay.append(file_picker)

    select_image = TextButton(content=initial_image, on_click=lambda _: file_picker.pick_files(allow_multiple=False))

    prompt_text = Text("", font_family="Minecraft", size=16, weight="bold", color="#000000")
    #prompt_input = ft.TextField(width=350,)

    #prompt_display = Text("LLM will respond here.", size=16, font_family="RobotoMono", weight=FontWeight.W_300, color="#cbddd1")
    prompt_display = Container(
        height = 450,
        content=ft.Column(scroll="auto")
    )

    prompt_container = Container(content=ft.Column([prompt_text, prompt_display]), visible=False)

    def route_change(e):
        page.views.clear()

        page.views.append(
            View(
                "/",
                [
                    AppBar(title_spacing=50, title=Text("OCR", font_family="Minecraft", size=40, weight=FontWeight.W_700, color="#000000"), bgcolor="#fffff", toolbar_height=120,
                           actions=[
                               TextButton(content=Container(Text("GitHub", font_family="Minecraft", size=18, weight=FontWeight.W_400, color="#000000"), padding=padding.only(right=25, left=25)), url="https://github.com/rjldg/OCR_project"),
                               TextButton(content=Container(Text("About Us", font_family="Minecraft", size=18, weight=FontWeight.W_400, color="#000000"), padding=padding.only(right=25, left=25)), on_click=lambda _: page.go("/aboutus"))
                           ]
                    ),
                    Container(
                        content=ft.Column(
                            [
                                Text("IMAGE-TO-TEXT APPLICATION\nUSING OPTICAL CHARACTER RECOGNITION", font_family="Minecraft", size=40, weight=FontWeight.W_800, color="#000000"),
                                Text("GREATEST PROJECT EVER MADE.", font_family="Minecraft", size=20, weight=FontWeight.W_500, color="#000000"),
                                Text("CPE179P - ARTIFICIAL INTELLIGENCE", font_family="Minecraft", size=14, weight=FontWeight.W_300, color="#000000"),
                            ]
                        ),
                        padding=padding.only(left=100, top=70)
                    ),
                    Container(
                        ft.ElevatedButton(content=Text("Get Started", font_family="Minecraft", size=18, weight=FontWeight.W_500, color="#000000"), on_click=lambda _: page.go("/ocr")),
                        padding=padding.only(left=100, top=70),
                    )
                ],
            )
        )
        
        if page.route == "/ocr":
            page.views.append(
                View(
                    "/ocr",
                    [
                        AppBar(color="#ffffff", bgcolor="#000000"),
                        Container(
                            content=ft.Row(
                                [
                                    Container(
                                        content=ft.Column(
                                            [
                                                result_image,
                                                restart_button,
                                            ],
                                            alignment=ft.MainAxisAlignment.CENTER,
                                        ),
                                        width=400,
                                        padding=padding.only(left=50, top=25)
                                    ),
                                    Container(
                                        select_image,
                                        padding=padding.only(top=100)
                                    ),
                                    Container(
                                        content=prompt_container,
                                        width=500,
                                        padding=padding.only(left=25, top=50)
                                    ),
                                ],
                                alignment=ft.MainAxisAlignment.CENTER,
                                spacing=50
                            ),
                        ),
                    ],
                )
            )
        if page.route == "/aboutus":
            page.views.append(
                View(
                    "/aboutus",
                    [
                        AppBar(color="#ffffff", bgcolor="#000000"),
                        Container(
                            Text("MEET THE TEAM", 
                                font_family="Minecraft", 
                                size=40, 
                                weight=FontWeight.W_700, 
                                color="#000000"),
                            padding=padding.only(left=100)
                        ),
                        Container(
                            content=ft.Row(
                                [
                                    ft.Column(
                                        [
                                            ft.Row(
                                                [
                                                    ft.Image(src="assets/aaron.png"),
                                                    Container(
                                                        content=ft.Column(
                                                            [
                                                                Text("Aaron Abadiano", 
                                                                    font_family="Minecraft", 
                                                                    size=20, 
                                                                    weight=FontWeight.W_600, 
                                                                    color="#000000"),
                                                                Text("ababadiano@mymail.mapua.edu.ph", 
                                                                    font_family="Minecraft", 
                                                                    size=16, 
                                                                    weight=FontWeight.W_300, 
                                                                    color="#000000"),
                                                            ]
                                                        )
                                                    ),
                                                ]
                                            ),

                                            ft.Row(
                                                [
                                                    ft.Image(src="assets/aaron.png"),
                                                    Container(
                                                        content=ft.Column(
                                                            [
                                                                Text("Azrael Anonas", 
                                                                    font_family="Minecraft", 
                                                                    size=20, 
                                                                    weight=FontWeight.W_600, 
                                                                    color="#000000"),
                                                                Text("@mymail.mapua.edu.ph", 
                                                                    font_family="Minecraft", 
                                                                    size=16, 
                                                                    weight=FontWeight.W_300, 
                                                                    color="#000000"),
                                                            ]
                                                        )
                                                    ),
                                                ]
                                            ),

                                            ft.Row(
                                                [
                                                    ft.Image(src="assets/rance_new.png"),
                                                    Container(
                                                        content=ft.Column(
                                                            [
                                                                Text("Rance De Guzman", 
                                                                    font_family="Minecraft", 
                                                                    size=20, 
                                                                    weight=FontWeight.W_600, 
                                                                    color="#000000"),
                                                                Text("rjldeguzman@mymail.mapua.edu.ph", 
                                                                    font_family="Minecraft", 
                                                                    size=16, 
                                                                    weight=FontWeight.W_300, 
                                                                    color="#000000"),
                                                            ]
                                                        )
                                                    ),
                                                ]
                                            ),
                                        ],
                                        spacing=20
                                    ),

                                    ft.Column(
                                        [
                                            ft.Row(
                                                [
                                                    ft.Image(src="assets/aaron.png"),
                                                    Container(
                                                        content=ft.Column(
                                                            [
                                                                Text("Angela Noelle Diala", 
                                                                    font_family="Minecraft", 
                                                                    size=20, 
                                                                    weight=FontWeight.W_600, 
                                                                    color="#000000"),
                                                                Text("@mymail.mapua.edu.ph", 
                                                                    font_family="Minecraft", 
                                                                    size=16, 
                                                                    weight=FontWeight.W_300, 
                                                                    color="#000000"),
                                                            ]
                                                        )
                                                    ),
                                                ]
                                            ),
                                            ft.Row(
                                                [
                                                    ft.Image(src="assets/aaron.png"),
                                                    Container(
                                                        content=ft.Column(
                                                            [
                                                                Text("Mikhaela Joie Dingrat", 
                                                                    font_family="Minecraft", 
                                                                    size=20, 
                                                                    weight=FontWeight.W_600, 
                                                                    color="#000000"),
                                                                Text("@mymail.mapua.edu.ph", 
                                                                    font_family="Minecraft", 
                                                                    size=16, 
                                                                    weight=FontWeight.W_300, 
                                                                    color="#000000"),
                                                            ]
                                                        )
                                                    ),
                                                ]
                                            ),
                                            ft.Row(
                                                [
                                                    ft.Image(src="assets/aaron.png"),
                                                    Container(
                                                        content=ft.Column(
                                                            [
                                                                Text("Nikhil Sahijwani", 
                                                                    font_family="Minecraft", 
                                                                    size=20, 
                                                                    weight=FontWeight.W_600, 
                                                                    color="#000000"),
                                                                Text("@mymail.mapua.edu.ph", 
                                                                    font_family="Minecraft", 
                                                                    size=16, 
                                                                    weight=FontWeight.W_300, 
                                                                    color="#000000"),
                                                            ]
                                                        )
                                                    ),
                                                ]
                                            ),
                                        ],
                                        spacing=20
                                    ),
                                ],
                                alignment=ft.MainAxisAlignment.SPACE_AROUND,
                            ),
                            padding=padding.symmetric(horizontal=50),
                        ),
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

ft.app(main)