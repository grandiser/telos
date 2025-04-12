#! /usr/bin/env python3
#  -*- coding: utf-8 -*-
#
# Combined PAGE GUI with custom logic (no UI_support)

import tkinter as tk
import tkinter.ttk as ttk
from tkinter.constants import *
from PIL import Image, ImageTk
import os.path

import PresentLabel

# Load functions
def load():
    import FileLoad3D
    FileLoad3D.main()

def load_nvidia():
    import FileLoad3Dgpu
    FileLoad3Dgpu.main()

def load_amd():
    import FileLoad3DAMDgpu
    FileLoad3DAMDgpu.main()

def presentlabel():
    PresentLabel.main()

# Optional style loader (from PAGE)
def _style_code():
    try:
        location = os.path.dirname(__file__)
        root.tk.call('source', os.path.join(location, 'themes', 'vista.tcl'))
    except Exception:
        pass
    style = ttk.Style()
    style.theme_use('vista')
    style.configure('.', font="TkDefaultFont")

class Toplevel1:
    def __init__(self, top=None):
        '''This class configures and populates the toplevel window.'''

        top.geometry("500x500+200+100")
        top.minsize(600, 400)
        top.maxsize(600, 400)
        top.title("Blood Vessel Labelling Tool")
        top.configure(highlightcolor="SystemWindowText")

        self.top = top

        #background image
        # Background image canvas
        self.Canvas1 = tk.Canvas(self.top, width=500, height=500)
        self.Canvas1.place(x=0, y=0, relwidth=1, relheight=1)

        # Use os.path.join to create a cross-platform file path
        bg_image_path = os.path.join(os.path.dirname(__file__), "bg.png")

        try:
            bg_image = Image.open(bg_image_path)
            bg_photo = ImageTk.PhotoImage(bg_image)
        except FileNotFoundError:
            print(f"File not found: {bg_image_path}")
        except Exception as e:
            print(f"An error occurred: {e}")

        self.Canvas1.create_image(0, 0, anchor='nw', image=bg_photo)
        self.bg_photo = bg_photo  # keep a reference to prevent GC
        

        # Title Label
        self.Label1 = tk.Label(self.top)
        self.Label1.place(relx=0.05, rely=0.05, height=52, relwidth=0.9)
        self.Label1.configure(
            font="-family {Segoe UI} -size 12 -weight bold",
            text="Machine Learning for Identifying Blood Vessels",
            relief="groove"
        )

        # Zarr Conversion Buttons Frame
        self.Frame1 = tk.Frame(self.top)
        self.Frame1.place(relx=0.05, rely=0.25, relheight=0.35, relwidth=0.9)
        self.Frame1.configure(relief='groove', borderwidth="2")

        # Preprocessing label
        self.Label2 = tk.Label(self.top)
        self.Label2.place(relx=0.1, rely=0.3, height=33, relwidth=0.8)
        self.Label2.configure(
            font="-family {Segoe UI} -size 10 -slant italic",
            text="Preprocessing: Convert TIFF to Zarr and expand feature set"
        )

        # Buttons
        self.TButton1 = ttk.Button(self.Frame1, text="CPU", command=load)
        self.TButton1.place(relx=0.05, rely=0.5, height=27, relwidth=0.25)

        self.TButton2 = ttk.Button(self.Frame1, text="NVIDIA GPU", command=load_nvidia)
        self.TButton2.place(relx=0.375, rely=0.5, height=27, relwidth=0.25)

        self.TButton3 = ttk.Button(self.Frame1, text="AMD GPU", command=load_amd)
        self.TButton3.place(relx=0.7, rely=0.5, height=27, relwidth=0.25)

        # Frame and Button for Zarr loading
        self.Frame2 = tk.Frame(self.top)
        self.Frame2.place(relx=0.05, rely=0.7, relheight=0.2, relwidth=0.9)
        self.Frame2.configure(relief='groove', borderwidth="2")

        self.Label2_1 = tk.Label(self.top)
        self.Label2_1.place(relx=0.1, rely=0.715, height=33, relwidth=0.8 )
        self.Label2_1.configure(
            font="-family {Segoe UI} -size 10 -slant italic",
            text="Semi-supervised prediction: Open Napari viewer"
        )

        self.TButton1_1 = ttk.Button(self.top, text="Napari", command=presentlabel)
        self.TButton1_1.place(relx=0.375, rely=0.80, height=27, relwidth=0.25)

# Entry point
def main():
    global root
    root = tk.Tk()
    _style_code()
    app = Toplevel1(top=root)
    root.mainloop()

if __name__ == '__main__':
    main()
