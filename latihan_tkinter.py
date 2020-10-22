import tkinter as tk
from PIL import Image, ImageTk
import tkinter.filedialog as filedialog
import cv2 as cv


def select_image():
    global panelA, panelB

    path = filedialog.askopenfilename()

    # ensure a file path was selected
    if len(path) > 0:
        # load the image from disk, convert it to grayscale
        img = cv.imread(path)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # OpenCV represents images in BGR order; however PIL represents
        # images in RGB order, so we need to swap the channels
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # convert the images to PIL format...
        img = Image.fromarray(img)
        img_gray = Image.fromarray(img_gray)

        # ...and then to ImageTk format
        img = ImageTk.PhotoImage(img)
        img_gray = ImageTk.PhotoImage(img_gray)

        # if the panels are None, initialize them
        if panelA is None or panelB is None:
            panelA = tk.Label(image=img)
            panelA.image = img
            panelA.pack(side="left", padx=10, pady=10)

            # panelB = tk.Label(image=img_gray)
            # panelB.image = img_gray
            # panelB.pack(side="right", padx=10, pady=10)


root = tk.Tk()
panelA = None
panelB = None

btn = tk.Button(root, text="Select an image", command=select_image)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

root.mainloop()
