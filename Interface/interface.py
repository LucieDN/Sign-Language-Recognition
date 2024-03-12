# hello_psg.py

import PySimpleGUI as sg
import cv2

# file_list_column = [
#     [
#         sg.Text("Image Folder"),
#         sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
#         sg.FolderBrowse(),
#     ],
#     [
#         sg.Listbox(
#             values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
#         )
#     ],
# ]

# image_viewer_column = [
#     [sg.Text("Choose an image from list on left:")],
#     [sg.Text(size=(40, 1), key="-TOUT-")],
#     [sg.Image(key="-IMAGE-")],
# ]

# video_viewer = [
#     [sg.Text("Choose an image from list on left:")],
#     [sg.Text(size=(40, 1), key="-TOUT-")],
#     [sg.Image(key="-IMAGE-")],
# ]
import PySimpleGUI as sg
import cv2
import numpy as np
import sys
import dearpygui.dearpygui as dpg

fn = sys.argv[1]
def load_video(fn):
    video = cv2.open(fn)
    fmt = 'rgb24'
    for f in video.decode():
        cf = f.to_ndarray(format=fmt)  # convert to rgb
        yield cf
    video.close()

video_gen = load_video(fn)
for f in video_gen:
    if dpg.is_dearpygui_running():
        update_dynamic_texture(f)
        dpg.render_dearpygui_frame()
    else:
        break  # deal with this scenario appropriately
dpg.destroy_context()


layout = [[sg.Text("Sign language dictionnary")],
        [sg.Button("Quitter")]]

# Create the window
window = sg.Window("Sign language dictionnary", layout,size=(1000, 600))

# window.set_options(size=(400, 200))

# Create an event loop
while True:
    event, values = window.read()
    # End program if user closes window or
    # presses the OK button
        
    if event == "Quitter" or event == sg.WIN_CLOSED:
        break
    

window.close()
