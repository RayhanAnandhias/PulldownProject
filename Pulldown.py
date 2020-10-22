import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import tkinter.filedialog as filedialog
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


class Pulldown(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.panel = None
        self.parent = parent
        self.file_menu = None
        self.mode_menu = None
        self.img_pil_format = None
        self.img_path = None
        self.imgcv= None
        self.title_base = self.parent.title()
        self.initialize_pulldown()

    def initialize_pulldown(self):
        menubar = tk.Menu(self.parent)

        # Adding File Menu and commands
        self.file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label='File', menu=self.file_menu)
        self.file_menu.add_command(label='Open', command=self.open_image)
        self.file_menu.add_command(label='Save', command=self.save_image, state="disabled")
        self.file_menu.add_command(label='Save As', command=self.save_as_image, state="disabled")
        self.file_menu.add_separator()
        self.file_menu.add_command(label='Exit', command=self.exit)

        # Adding mode color,grayscale
        self.mode_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label='Edit', menu=self.mode_menu)

        self.mode_menu.add_command(label='Original',
                                   command=lambda: self.open_image_bg(img_path=self.img_path),
                                   state="disabled")
        self.mode_menu.add_command(label='Grayscale',
                                   command=lambda: self.open_image_bg(img_path=self.img_path, mode=cv.COLOR_BGR2GRAY),
                                   state="disabled")
        self.mode_menu.add_command(label='2D Convolution', command=self.twoDConvolution, state="disabled")
        self.mode_menu.add_command(label='Averaging', command=self.averaging, state="disabled")
        self.mode_menu.add_command(label='Gaussian Filtering', command=self.gaussian, state="disabled")
        self.mode_menu.add_command(label='Median Filtering', command=self.median, state="disabled")
        self.mode_menu.add_command(label='Bilateral Filtering', command=self.bilateral, state="disabled")

        # display Menu
        self.parent.config(menu=menubar)

    def open_image(self):
        # Load Image Path
        self.img_path = filedialog.askopenfilename(
                title="Open Image",
                filetypes=[('image files', ('.png', '.jpg', '.jpeg'))])

        if len(self.img_path) > 0:
            # load the image from disk
            img = cv.imread(self.img_path)
            self.imgcv = img

            # OpenCV represents images in BGR order; however PIL represents
            # images in RGB order, so we need to swap the channels
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            # convert the images to PIL format...
            img = Image.fromarray(img)
            self.img_pil_format = img

            # ...and then to ImageTk format
            img = ImageTk.PhotoImage(img)

            # change window size
            self.parent.geometry("{width}x{height}".format(width=img.width(), height=img.height()))

            # if the panels are None, initialize them
            if self.panel is None:
                self.panel = tk.Label(image=img)
                self.panel.image = img
            else:
                self.panel.configure(image=img)
                self.panel.image = img

            self.panel.pack()
            self.file_menu.entryconfig("Save As", state="normal")
            self.file_menu.entryconfig("Save", state="normal")
            self.mode_menu.entryconfig("Original", state="normal")
            self.mode_menu.entryconfig("Grayscale", state="normal")
            self.mode_menu.entryconfig("2D Convolution", state="normal")
            self.mode_menu.entryconfig("Averaging", state="normal")
            self.mode_menu.entryconfig("Gaussian Filtering", state="normal")
            self.mode_menu.entryconfig("Median Filtering", state="normal")
            self.mode_menu.entryconfig("Bilateral Filtering", state="normal")
            self.parent.title(self.title_base + " - " + self.img_path)

    def open_image_bg(self, img_path, mode=cv.COLOR_BGR2RGB):
        # load the image from disk
        img = cv.imread(img_path)
        self.imgcv = img

        # OpenCV represents images in BGR order; however PIL represents
        # images in RGB order, so we need to swap the channels
        img = cv.cvtColor(img, mode)

        # convert the images to PIL format...
        img_cv_pil = Image.fromarray(img)
        self.img_pil_format = img_cv_pil

        # ...and then to ImageTk format
        img_cv_pil = ImageTk.PhotoImage(img_cv_pil)

        # change window size
        self.parent.geometry("{width}x{height}".format(width=img_cv_pil.width(), height=img_cv_pil.height()))

        self.panel.configure(image=img_cv_pil)
        self.panel.image = img_cv_pil
        self.panel.pack()
        self.parent.title(self.title_base + " - " + self.img_path)

        if mode == cv.COLOR_BGR2GRAY:
            plt.subplot(121), plt.imshow(cv.cvtColor(self.imgcv, cv.COLOR_BGR2RGB)), plt.title('Original')
            plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(img, cmap="gray"), plt.title('Grayscale')
            plt.xticks([]), plt.yticks([])
            plt.show()

    def save_image(self):
        if self.panel is not None:
            self.img_pil_format.save(self.img_path)
            messagebox.showinfo("showinfo", "Image Saved")
        else:
            return

    def save_as_image(self):
        if self.panel is not None:
            filetypes = [('PNG', '*.png'), ('JPG', '*.jpg'), ('JPEG', '*.jpeg')]
            fname = filedialog.asksaveasfilename(
                    title="Save As Image",
                    filetypes=filetypes,
                    defaultextension=filetypes)
            if not fname:
                return

            self.img_pil_format.save(fname)
            self.img_path = fname
            self.open_image_bg(img_path=self.img_path)
        else:
            return

    def twoDConvolution(self):
        img_cv = self.imgcv
        img_cv = cv.cvtColor(img_cv, cv.COLOR_BGR2RGB)

        kernel = np.ones((5,5), np.float32)/25
        dst = cv.filter2D(img_cv, -1, kernel)

        # convert the images to PIL format...
        img_cv_pil = Image.fromarray(dst)
        self.img_pil_format = img_cv_pil

        # ...and then to ImageTk format
        img_cv_pil = ImageTk.PhotoImage(img_cv_pil)

        # change window size
        self.parent.geometry("{width}x{height}".format(width=img_cv_pil.width(), height=img_cv_pil.height()))

        self.panel.configure(image=img_cv_pil)
        self.panel.image = img_cv_pil
        self.panel.pack()
        self.parent.title(self.title_base + " - " + self.img_path)

        plt.subplot(121), plt.imshow(img_cv), plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(dst), plt.title('2D Convolution')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def averaging(self):
        img_cv = self.imgcv
        img_cv = cv.cvtColor(img_cv, cv.COLOR_BGR2RGB)

        dst = cv.blur(img_cv,(5,5))

        # convert the images to PIL format...
        img_cv_pil = Image.fromarray(dst)
        self.img_pil_format = img_cv_pil

        # ...and then to ImageTk format
        img_cv_pil = ImageTk.PhotoImage(img_cv_pil)

        # change window size
        self.parent.geometry("{width}x{height}".format(width=img_cv_pil.width(), height=img_cv_pil.height()))

        self.panel.configure(image=img_cv_pil)
        self.panel.image = img_cv_pil
        self.panel.pack()
        self.parent.title(self.title_base + " - " + self.img_path)

        plt.subplot(121), plt.imshow(img_cv), plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def gaussian(self):
        img_cv = self.imgcv
        img_cv = cv.cvtColor(img_cv, cv.COLOR_BGR2RGB)

        dst = cv.GaussianBlur(img_cv,(5,5),0)

        # convert the images to PIL format...
        img_cv_pil = Image.fromarray(dst)
        self.img_pil_format = img_cv_pil

        # ...and then to ImageTk format
        img_cv_pil = ImageTk.PhotoImage(img_cv_pil)

        # change window size
        self.parent.geometry("{width}x{height}".format(width=img_cv_pil.width(), height=img_cv_pil.height()))

        self.panel.configure(image=img_cv_pil)
        self.panel.image = img_cv_pil
        self.panel.pack()
        self.parent.title(self.title_base + " - " + self.img_path)

        plt.subplot(121), plt.imshow(img_cv), plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(dst), plt.title('Gaussian Filtering')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def median(self):
        img_cv = self.imgcv
        img_cv = cv.cvtColor(img_cv, cv.COLOR_BGR2RGB)

        dst = cv.medianBlur(img_cv,5)

        # convert the images to PIL format...
        img_cv_pil = Image.fromarray(dst)
        self.img_pil_format = img_cv_pil

        # ...and then to ImageTk format
        img_cv_pil = ImageTk.PhotoImage(img_cv_pil)

        # change window size
        self.parent.geometry("{width}x{height}".format(width=img_cv_pil.width(), height=img_cv_pil.height()))

        self.panel.configure(image=img_cv_pil)
        self.panel.image = img_cv_pil
        self.panel.pack()
        self.parent.title(self.title_base + " - " + self.img_path)

        plt.subplot(121), plt.imshow(img_cv), plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(dst), plt.title('Median Filtering')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def bilateral(self):
        img_cv = self.imgcv
        img_cv = cv.cvtColor(img_cv, cv.COLOR_BGR2RGB)

        dst = cv.bilateralFilter(img_cv,9,75,75)

        # convert the images to PIL format...
        img_cv_pil = Image.fromarray(dst)
        self.img_pil_format = img_cv_pil

        # ...and then to ImageTk format
        img_cv_pil = ImageTk.PhotoImage(img_cv_pil)

        # change window size
        self.parent.geometry("{width}x{height}".format(width=img_cv_pil.width(), height=img_cv_pil.height()))

        self.panel.configure(image=img_cv_pil)
        self.panel.image = img_cv_pil
        self.panel.pack()
        self.parent.title(self.title_base + " - " + self.img_path)

        plt.subplot(121), plt.imshow(img_cv), plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(dst), plt.title('Bilateral Filtering')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def exit(self):
        var = messagebox.askyesno("Exit", "Do you want to exit ?")
        if var is True:
            self.parent.destroy()
        else:
            return
