import tkinter as tk


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

    def cleanup(self):
        self.value=self.e.get()
        self.top.destroy()