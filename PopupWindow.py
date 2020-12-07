import tkinter as tk
from tkinter import messagebox
import fractions


class PopupWindow(object):
    def __init__(self,master):
        self.top=tk.Toplevel(master)

    def basic_popup(self):
        self.top.geometry("300x100")
        self.top.title("Enter Kernel size (nxn)")
        l = tk.Label(self.top, text="Enter n : ")
        l.pack()
        self.e = tk.Entry(self.top)
        self.e.pack()
        b = tk.Button(self.top, text='Submit', command=self.cleanup)
        b.pack()

    def input_matrix(self, rows, cols, mx=None):
        self.top.geometry("400x300")
        self.top.title("Kernel")

        # empty arrays for your Entrys and StringVars
        text_var = []
        entries = []

        tk.Label(self.top, text="Enter matrix :").place(x=20, y=20)
        x2 = 0
        y2 = 0

        for i in range(rows):
            # append an empty list to your two arrays
            # so you can append to those later
            text_var.append([])
            entries.append([])
            for j in range(cols):
                # append your StringVar and Entry
                if mx is not None :
                    text_var[i].append(tk.StringVar(value=mx[i][j]))
                else:
                    text_var[i].append(tk.StringVar())
                entries[i].append(tk.Entry(self.top, textvariable=text_var[i][j], width=4))
                entries[i][j].place(x=60 + x2, y=50 + y2)
                x2 += 30
            y2 += 30
            x2 = 0

        b = tk.Button(self.top, text="Submit", width=15, command=lambda: self.get_mat(rows, cols, text_var))
        b.place(x=160, y=200)

    def get_mat(self, rows, cols, text_var):
        try:
            matrix = []
            for i in range(rows):
                matrix.append([])
                for j in range(cols):
                    matrix[i].append(float(fractions.Fraction(text_var[i][j].get())))
            print(matrix)
            self.kernel = matrix
            self.top.destroy()
        except Exception as e :
            # print(e)
            self.top.destroy()

    def input_bilateral(self):
        self.top.geometry("400x300")
        self.top.title("Enter Parameter for Bilateral Filtering")
        l = tk.Label(self.top, text="Enter diameter : ")
        l.pack()
        self.e1 = tk.Entry(self.top)
        self.e1.insert(0, "9") #default value
        self.e1.pack()
        l1 = tk.Label(self.top, text="Enter sigmaColor : ")
        l1.pack()
        self.e2 = tk.Entry(self.top)
        self.e2.insert(0, "75") #default value
        self.e2.pack()
        l2 = tk.Label(self.top, text="Enter sigmaSpace : ")
        l2.pack()
        self.e3 = tk.Entry(self.top)
        self.e3.insert(0, "75") #default value
        self.e3.pack()
        b = tk.Button(self.top, text='Submit', command=self.cleanup_bilateral)
        b.pack()

    def cleanup_bilateral(self):
        self.diameter=self.e1.get()
        self.sigmaColor=self.e2.get()
        self.sigmaSpace=self.e3.get()
        self.top.destroy()

    def input_threshold(self):
        self.top.geometry("400x300")
        self.top.title("Enter Parameter for Image Thresholding")
        l = tk.Label(self.top, text="Enter Threshold value : ")
        l.pack()
        self.e1 = tk.Entry(self.top)
        self.e1.insert(0, "127")  # default value
        self.e1.pack()
        l1 = tk.Label(self.top, text="Enter maxval : ")
        l1.pack()
        self.e2 = tk.Entry(self.top)
        self.e2.insert(0, "255")  # default value
        self.e2.pack()
        b = tk.Button(self.top, text='Submit', command=self.cleanup_threshold)
        b.pack()

    def cleanup_threshold(self):
        self.threshold = self.e1.get()
        self.maxval = self.e2.get()
        self.top.destroy()

    def input_salt(self):
        self.top.geometry("300x100")
        self.top.title("Enter Parameter Salt and Pepper Noise")
        l = tk.Label(self.top, text="Enter amount : ")
        l.pack()
        self.e = tk.Entry(self.top)
        self.e.pack()
        b = tk.Button(self.top, text='Submit', command=self.cleanup)
        b.pack()

    def input_gauss_speckle(self):
        self.top.geometry("300x100")
        self.top.title("Enter Parameter Gaussian Noise")
        l = tk.Label(self.top, text="Enter Mean : ")
        l.pack()
        self.e = tk.Entry(self.top)
        self.e.insert(0, "0")
        self.e.pack()
        l1 = tk.Label(self.top, text="Enter Variance : ")
        l1.pack()
        self.e1 = tk.Entry(self.top)
        self.e1.insert(0, "0.05")
        self.e1.pack()
        b = tk.Button(self.top, text='Submit', command=self.cleanup_gaussspeckle_noise)
        b.pack()

    def cleanup_gaussspeckle_noise(self):
        self.mean=self.e.get()
        self.variance = self.e1.get()
        self.top.destroy()

    def cleanup(self):
        self.value=self.e.get()
        self.top.destroy()
