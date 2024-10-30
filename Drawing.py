########### Imports ###########

from tkinter import Tk, Frame, Canvas, CENTER, Button, NW, Label, SOLID
from tkinter import colorchooser, filedialog, OptionMenu, messagebox
from tkinter import DOTBOX, StringVar, simpledialog
from tkinter import *

import os
import pickle
import mic, model
##Need to increase/decrease erase size
#Need to

########### Window Settings ###########

## 

root = Tk()
canvas = Canvas(root)

root.title("Paint - ECTS v1.0")
root.geometry("650x650")

root.resizable(False, False)

########### Functions ###########

# Variables
prevPoint = [0, 0]
currentPoint = [0, 0]
penColor = "black"
stroke = 1
isLasso = False
lassoId = 0
lassoStartX = 0
lassoStartY = 0
lassoEndX = 0
lassoEndY = 0
moveLasso = False
lassoObjects = []

canvas_data = []

shapeSelect = StringVar()
shapeList = ["None", "Square", "Circle/Oval", "Rectangle", "Line"]
shapeSelect.set("None")
selected_shape = None
shapes = []
shapeFill = "black"
width = 0
height = 0

# Increase Stroke Size By 1
def strokeI():
    global stroke

    if stroke != 10:
        stroke += 1

    else:
        stroke = stroke


# Decrease Stroke Size By 1
def strokeD():
    global stroke

    if stroke != 1:
        stroke -= 1

    else:
        stroke = stroke


def strokeDf():
    global stroke
    stroke = 1

# Pencil
def pencil():
    global penColor

    resetLasso()

    penColor = "black"
    canvas["cursor"] = "pencil"

# Lasso
def lasso():
    global penColor
    global isLasso
    global moveLasso
    global lassoId

    resetLasso()

    isLasso = True

    penColor = "red"
    canvas["cursor"] = "crosshair"


def resetLasso():

    global lassoEndX
    global lassoEndY
    global lassoStartX
    global lassoStartY
    global moveLasso
    global lassoId
    global lassoObjects
    global isLasso

    lassoEndX = 0
    lassoEndY = 0
    lassoStartX = 0
    lassoStartY = 0

    isLasso = False
    moveLasso = False
    canvas.delete(lassoId)
    lassoObjects = []

# Eraser is less erasing and actually just painting over the pencil with white
def eraser():
    global penColor

    penColor = "white"
    canvas["cursor"] = DOTBOX


# Pencil Choose Color
def colorChoice():
    global penColor

    color = colorchooser.askcolor(title="Select a Color")
    canvas["cursor"] = "pencil"

    if color[1]:
        penColor = color[1]

    else:
        pass


# Shape Color Chooser
def shapeColorChoice():
    global shapeFill

    color = colorchooser.askcolor(title="Select a Color")
    canvas["cursor"] = "pencil"

    if color[1]:
        shapeFill = color[1]

    else:
        shapeFill = "black"

def on_shape_click(event):
    #Capture the initial position of the mouse click
    global currentPoint, shapeSelect, shapeFill, selected_shape, width, height

    x = event.x
    y = event.y

    # Check if a shape is selected in the OptionMenu
    if shapeSelect.get() != "None":
        askShapeDimension()  # Ask for dimensions if needed

        # Create the shape based on the selection
        match shapeSelect.get():
            case "Square":
                canvas.create_rectangle(x, y, x + width, y + height, fill=shapeFill)
            case "Circle/Oval":
                canvas.create_oval(x, y, x + width, y + height, fill=shapeFill)
            case "Rectangle":
                canvas.create_rectangle(x, y, x + width, y + height, fill=shapeFill)
            case "Line":
                canvas.create_line(x, y, x + width, y + height, fill=shapeFill, width=stroke)
            case _:
                pass
    else:
        # If no shape is selected, allow the user to move shapes as usual
        shape = canvas.find_closest(x, y)
        selected_shape = shape[0]

        currentPoint = [x, y]

def on_shape_drag(event):
    global currentPoint
    global selected_shape

    x = currentPoint[0]
    y = currentPoint[1]

    if selected_shape is not None:
        # Calculate the difference in position
        dx = event.x - x
        dy = event.y - y

        # Move the shape by that distance
        canvas.move(selected_shape, dx, dy)

        # Update the known position
        x = event.x
        y = event.y
        currentPoint = [x,y]

# Paint Function
def paint(event):
    global prevPoint
    global currentPoint
    global moveLasso
    global isLasso

    x = event.x
    y = event.y

    currentPoint = [x, y]

    if moveLasso:
        moveLassoObject("none")
    if isLasso:
        createLasso(event)
    elif prevPoint != [0, 0]:
        canvas.create_polygon(
            prevPoint[0],
            prevPoint[1],
            currentPoint[0],
            currentPoint[1],
            fill=penColor,
            outline=penColor,
            width=stroke,
        )

    prevPoint = currentPoint

    if event.type == "5":
        prevPoint = [0, 0]

def createLasso(event):
    global lassoId
    global lassoStartX
    global lassoStartY
    global lassoEndX
    global lassoEndY
    global moveLasso
    global lassoObjects

    if not moveLasso:
        if str(event.type) == "4":
            lassoStartX = event.x
            lassoStartY = event.y
        elif str(event.type) == "5":
            lassoEndX = event.x
            lassoEndY = event.y
            lassoId = canvas.create_rectangle(
                    lassoStartX,
                    lassoStartY,
                    lassoEndX,
                    lassoEndY,
                    fill='',
                    dash= (5,),
                    outline=penColor,
                    width=stroke,
                )
            moveLasso = True

            if len(lassoObjects) == 0:
                lassoObjects = canvas.find_enclosed(canvas.bbox(lassoId)[0],
                                                    canvas.bbox(lassoId)[1],
                                                    canvas.bbox(lassoId)[2],
                                                    canvas.bbox(lassoId)[3])

def moveLassoObject(direction):

    global lassoId
    global lassoObjects

    translate = []

    if direction == "Left":
        translate = [-1, 0]
    elif direction == "Right":
        translate = [1, 0]
    elif direction == "Up":
        translate = [0, -1]
    elif direction == "Down":
        translate = [0, 1]
    else:
        return


    for object in lassoObjects:
        canvas.move(object, translate[0], translate[1])



    

# Close App
def newApp():
    os.startfile("paint.py")


# Clear Screen
def clearScreen():
    canvas.delete("all")


# Save Images
def saveImg():
    global canvas_data
    for obj in canvas.find_all():
        obj_type = canvas.type(obj)
        if obj_type == "polygon":
            color = canvas.itemcget(obj, "fill")
            coords = canvas.coords(obj)
            canvas_data.append({"type": "polygon", "color": color, "coords": coords})

    saveEcts()


# Saving the canvas data to ects files
def saveEcts():
    global canvas_data
    file_path = filedialog.asksaveasfilename(
        defaultextension=".ects",
        filetypes=[
            ("ECTS files", "*.ects"),
            ("PNG files", "*.png"),
            ("JPG files", "*.jpg"),
        ],
    )
    if file_path:
        with open(file_path, "wb") as file:
            pickle.dump(canvas_data, file)


# Opening already or earlier made ects files
def openEcts():
    global canvas_data
    file_path = filedialog.askopenfilename(
        defaultextension=".ects",
        filetypes=[
            ("ECTS files", "*.ects"),
            ("PNG files", "*.png"),
            ("JPG files", "*.jpg"),
        ],
    )
    if file_path:
        with open(file_path, "rb") as file:
            canvas_data = pickle.load(file)

        redrawCanvas()


# Redrawing the Canvas Data after opening it
def redrawCanvas():
    global canvas_data
    # Clear the canvas
    canvas.delete("all")
    # Draw objects from canvas_data
    for obj in canvas_data:
        if obj["type"] == "polygon":
            color = obj["color"]
            coords = obj["coords"]
            canvas.create_polygon(coords, fill=color, outline=color, width=stroke)


# Asking Shape Dimensions
def askShapeDimension():
    global width, height

    width = simpledialog.askinteger(
        "ECTS - Paint App", f"Enter Width for {shapeSelect.get()}"
    )

    height = simpledialog.askinteger(
        "ECTS - Paint App", f"Enter Height for {shapeSelect.get()}"
    )
    if width and height:
        print(width, height)


# Key Binding to Show Shape OptionMenu
def show_shape_menu(event):
    # Get the current position of the shape OptionMenu
    x = shapeMenu.winfo_rootx()
    y = shapeMenu.winfo_rooty() + shapeMenu.winfo_height()

    # Post the menu at the current position
    shapeMenu['menu'].post(x, y)

def speak():
    # Run the drawing app with the retrieved image pixel data
    print("say your object please")
    user_speech = mic.talk()
    model.open_gallery_window(canvas, user_speech)


########### Paint App ###########

#### Paint Tools Frame ####

# Main Frame
frame1 = Frame(root, height=150, width=650)
frame1.grid(row=0, column=0)

# Holder Frame
holder = Frame(frame1, height=120, width=600, bg="white", pady=10)
holder.grid(row=0, column=0, sticky=W)
holder.place(x=0, y=0)

holder.columnconfigure(0, minsize=120)
holder.columnconfigure(1, minsize=120)
holder.columnconfigure(2, minsize=120)
holder.columnconfigure(3, minsize=120)
holder.columnconfigure(4, minsize=120)

holder.rowconfigure(0, minsize=30)

#### Tools ####

# Label for Tool 1,2,3
label123 = Label(holder, text="TOOLS", borderwidth=1, relief=SOLID, width=15)
label123.grid(row=0, column=0)

# Tool 1 - Pencil
pencilButton = Button(holder, text="Pencil", height=1, width=12, command=pencil)
pencilButton.grid(row=1, column=0)

# Tool 2 - Eraser
eraserButton = Button(holder, text="Eraser", height=1, width=12, command=eraser)
eraserButton.grid(row=2, column=0)

# Tool 3 - Color Change
colorButton = Button(
    holder, text="Select Color", height=1, width=12, command=colorChoice
)
colorButton.grid(row=3, column=0)

#### FILE ACTIONS ####

# Label for Tool 4,5,6
label456 = Label(holder, text="FILE", borderwidth=1, relief=SOLID, width=15)
label456.grid(row=0, column=1)

# Tool 4 - Save File
saveButton = Button(holder, text="SAVE", height=1, width=12, command=saveImg)
saveButton.grid(row=1, column=1)

# Tool 5 - Open File
openButton = Button(holder, text="OPEN", height=1, width=12, command=openEcts)
openButton.grid(row=2, column=1)

# Tool 6 - New Paint
newButton = Button(holder, text="NEW", height=1, width=12, command=newApp)
newButton.grid(row=3, column=1)

#### OTHER ####

# Label for Tool 7 and 8
label7 = Label(holder, text="OTHER", borderwidth=1, relief=SOLID, width=15)
label7.grid(row=0, column=2)

# Tool 7 - Clear Screen
clearButton = Button(holder, text="CLEAR", height=1, width=12, command=clearScreen)
clearButton.grid(row=1, column=2)

# Tool 8 - Lasso
lassoButton = Button(holder, text="LASSO", height=1, width=12, command=lasso)
lassoButton.grid(row=2, column=2)

# Tool 9 - Exit App
exitButton = Button(
    holder, text="Exit", height=1, width=12, command=lambda: root.destroy())
exitButton.grid(row=3, column=2)

#### Stroke Size ####

# Label for Tool 8, 9 and 10
label8910 = Label(holder, text="STROKE SIZE", borderwidth=1, relief=SOLID, width=15)
label8910.grid(row=0, column=3)

# Tool 8 - Increament by 1
sizeiButton = Button(holder, text="Increase", height=1, width=12, command=strokeI)
sizeiButton.grid(row=1, column=3)

# Tool 9 - Decreament by 1
sizedButton = Button(holder, text="Decrease", height=1, width=12, command=strokeD)
sizedButton.grid(row=2, column=3)

# Tool 10 - Default
defaultButton = Button(holder, text="Default", height=1, width=12, command=strokeDf)
defaultButton.grid(row=3, column=3)

#### Shapes ####

# Label for Tool 11,12,13
label1123 = Label(holder, text="SHAPES", borderwidth=1, relief=SOLID, width=15)
label1123.grid(row=0, column=4)

# Tool 11 - shapeSelector
shapeMenu = OptionMenu(holder, shapeSelect, *shapeList)
shapeMenu.grid(row=1, column=4)
shapeMenu.config(width=8)

# Tool 9 - Decreament by 1
DimensionButton = Button(
    holder, text="Dimension", height=1, width=12, command=askShapeDimension
)
DimensionButton.grid(row=2, column=4)

# Tool 9 - Speech-Draw
DimensionButton = Button(
    holder, text="speech-Draw", height=1, width=12, command=speak
)
DimensionButton.grid(row=3, column=4)

# Tool 10 - Default
fillButton = Button(holder, text="Fill", height=1, width=12, command=shapeColorChoice)
fillButton.grid(row=4, column=4)



#### Canvas Frame ####

# Main Frame
frame2 = Frame(root, height=500, width=650)
frame2.grid(row=1, column=0)

# Making a Canvas
canvas = Canvas(frame2, height=450, width=550, bg="white")
canvas.grid(row=0, column=0)
canvas.place(relx=0.5, rely=0.5, anchor=CENTER)
canvas.config(cursor="pencil")

# Event Binding
canvas.bind("<B1-Motion>", paint)
canvas.bind("<ButtonRelease-1>", paint)
canvas.bind("<Button-1>", paint)
canvas.bind("<Button-3>", on_shape_click)
canvas.bind("<B3-Motion>", on_shape_drag)


# Key Bindings
root.bind("<=>", lambda event: strokeI())
root.bind("<minus>", lambda event: strokeD())
root.bind("<p>", lambda event: pencil())
root.bind("<e>", lambda event: eraser())
root.bind("<c>", lambda event: colorChoice())
root.bind("<Control-s>", lambda event: saveImg())
root.bind("<Control-o>", lambda event: openEcts())
root.bind("<Control-n>", lambda event: newApp())
root.bind("<l>", lambda event: lasso())
root.bind("<Delete>", lambda event: clearScreen())
root.bind("<Control-d>", lambda event: clearScreen())
root.bind("<d>", lambda event: askShapeDimension())
root.bind("<f>", lambda event: shapeColorChoice())
root.bind("<t>", lambda event: speak())
root.bind("<s>", show_shape_menu)
root.bind("<l>", lambda event: lasso())
root.bind("<Left>", lambda event: moveLassoObject("Left"))
root.bind("<Right>", lambda event: moveLassoObject("Right"))
root.bind("<Up>", lambda event: moveLassoObject("Up"))
root.bind("<Down>", lambda event: moveLassoObject("Down"))

########### Main Loop ###########
canvas.pack()
root.mainloop()



