import os
import shutil
import tempfile
import flet as ft

def main(page: ft.Page):
    page.title = "PH Motor Plate OCR"
    page.window_width = 1024
    page.window_height = 768
    page.fonts = {
        "Century Gothic" : "assets/fonts/centurygothic.ttf",
        "Century Gothic Bold" : "assets/fonts/centurygothic_bold.ttf",
        "Adequate" : "assets/fonts/Adequate-ExtraLight.ttf"
    }
    page.theme = ft.Theme(font_family="centurygothic")
    page.horizontal_alignment=ft.CrossAxisAlignment.CENTER


    # COMPONENTS #
    title_text = ft.Text("PH Motor Plate OCR", font_family="Adequate")
    button_start = ft.ElevatedButton(text="Begin", bgcolor="#e9ff00", color="#2c2c37")
    button_about = ft.ElevatedButton(text="About", bgcolor="#2c2c37", color="#e9ff00")
    # USER CONTROLS #
    button_layout = ft.Row(controls=[button_start, button_about], alignment=ft.MainAxisAlignment.START)
    page_layout = ft.Column(controls=[title_text, button_layout])

    # LAYOUT # 

    
    page.add()

ft.app(target=main, assets_dir="assets")
