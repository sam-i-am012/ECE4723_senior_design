import tkinter as tk # gui library for python 

class InvisiVisorSim:
    # def __init__(self):
    def __init__(self, monitor_width=1920, monitor_height=1080, laptop_height=0, laptop_width=0):
        self.root = tk.Tk() # create main application window 

        # self.root.overrideredirect(True) # remove window borders and title bar

        # for second monitor 
        # self.root.geometry(f"{monitor_width}x{monitor_height}+{laptop_width}+{laptop_height}") # set the window size and position adjusted for monitor set up 
        
        self.root.attributes('-fullscreen', True) # full screen
        
        self.root.attributes('-topmost', True) # window will always be on top 
        self.root.config(cursor="none") # no mouse cursor 
        
        self.canvas = tk.Canvas(self.root, highlightthickness=0, bg="white") # the canvas is what we will draw on (white background for "transparency")
        self.canvas.pack(fill="both", expand=True)

        self.block_size = 150
        self.rect = self.canvas.create_rectangle(0, 0, self.block_size, self.block_size, fill="black") # create a black rectangle that will follow the mouse/eye movement
        
        self.root.bind('<Motion>', self.move_block) # bind mouse movement to the move_block function
        self.root.bind('<Escape>', lambda e: self.root.destroy()) # exit application when escape key is pressed

    def move_block(self, event): # this just centers the black box around the mouse but will change when we have actual eye tracking data
        x1 = event.x - (self.block_size / 2)
        y1 = event.y - (self.block_size / 2)
        x2 = event.x + (self.block_size / 2)
        y2 = event.y + (self.block_size / 2)
        
        self.canvas.coords(self.rect, x1, y1, x2, y2)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    # app = InvisiVisorSim()
    app = InvisiVisorSim(monitor_width=1920, monitor_height=1080, laptop_width=0, laptop_height=0) # adjust these values based on your monitor setup
    app.run()