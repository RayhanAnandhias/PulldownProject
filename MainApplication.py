import tkinter as tk
from Pulldown import Pulldown


class MainApplication(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.initialize_user_interface()
        self.pulldown = Pulldown(parent)

    def initialize_user_interface(self):
        self.parent.geometry("500x500")
        self.parent.title("[IP]181524028")


if __name__ == "__main__":
    root = tk.Tk()
    MainApplication(root).pack()
    root.mainloop()
