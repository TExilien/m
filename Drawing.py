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
root.geometry("800x800")

root.resizable(True, True)

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
inZoom = False
toolNames = ["Pencil", "Eraser", "Lasso"]
toolSelect = StringVar()
toolSelect.set("Pencil")

canvas_data = []

WIDTH = 80
HEIGHT = 450

fileSelect = StringVar()
fileList = ["None", "Open", "New"]



toolNumber = 0
numTools = 3

def cycleTool():
    global toolNumber, toolSelect
    toolNumber = (toolNumber + 1) % numTools
    toolSelect.set(toolNames[toolNumber])
    match toolNumber:
        case 0:
            pencil()
        case 1:
            lasso()
        case 2:
            eraser()


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

# Zoom

def zoomControl():
    global inZoom

    inZoom = True

# Pencil
def pencil():
    global penColor
    global inZoom

    resetLasso()
    inZoom = False

    penColor = "black"
    canvas["cursor"] = "pencil"

# Lasso
def lasso():
    global penColor
    global isLasso
    global moveLasso
    global lassoId
    global inZoom

    inZoom = False
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

    resetLasso()
    inZoom = False


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

# Paint Function
def paint(event):
    global prevPoint
    global currentPoint
    global moveLasso
    global isLasso

    x = event.x
    y = event.y

    currentPoint = [x, y]

    if inZoom:
        pass
    elif moveLasso:
        moveLassoObject("none")
    elif isLasso:
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
        translate = [(currentPoint[0]-prevPoint[0]), (currentPoint[1]-prevPoint[1])]


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
        
# Speech To Draw 
def speak():
    # Run the drawing app with the retrieved image pixel data
    print("say your object please")
    user_speech = mic.talk()
    model.open_gallery_window(canvas, user_speech)


# Zooming in and out of the canvas
def zoom(event, scale):

    canvas.scale("all", event.x, event.y, scale, scale)


########### Paint App ###########

#### Paint Tools Frame ####

# Main Frame
frame1 = Frame(root, height=150, width=800)
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
label123 = Label(holder, text="TOOLS", borderwidth=1, relief=SOLID, width=int(WIDTH/4))
label123.grid(row=0, column=0)



# Tool 1 - Cycle (Pencil, Eraser, Lasso)
toolMenu = OptionMenu(holder, toolSelect, *toolNames)
toolMenu.grid(row=1, column=0)
toolMenu.config(height=1, width=int(WIDTH/4 - 5))

'''
selectTool = Button(holder, text=toolSelect, height=1, width=12)
selectTool.grid(row=1, column=0)
'''

# Tool 2 - Color Change
colorButton = Button(
    holder, text="Select Color", height=1, width=int(WIDTH/4 - 3), command=colorChoice
)
colorButton.grid(row=2, column=0)

# Tool 3 - Exit App
exitButton = Button(
    holder, text="Exit", height=1, width=int(WIDTH/4 - 3), command=lambda: root.destroy())
exitButton.grid(row=3, column=0)


#### FILE ACTIONS ####

# Label for Tool 4,5,6 
label456 = Label(holder, text="FILE", borderwidth=1, relief=SOLID, width=int(WIDTH/4))
label456.grid(row=0, column=1)

# Tool 4 - Save File
saveButton = Button(holder, text="SAVE", height=1, width=int(WIDTH/4 - 3), command=saveImg)
saveButton.grid(row=1, column=1)

# Tool 5 - Open File
openButton = Button(holder, text="OPEN", height=1, width=int(WIDTH/4 - 3), command=openEcts)
openButton.grid(row=2, column=1)

# Tool 6 - New Paint
newButton = Button(holder, text="NEW", height=1, width=int(WIDTH/4 - 3), command=newApp)
newButton.grid(row=3, column=1)




#### OTHER ####

# Label for Tool 7, 8, 9
label7 = Label(holder, text="OTHER", borderwidth=1, relief=SOLID, width=int(WIDTH/4))
label7.grid(row=0, column=2)

# Tool 7 - Clear Screen
clearButton = Button(holder, text="CLEAR", height=1, width=int(WIDTH/4 - 3), command=clearScreen)
clearButton.grid(row=1, column=2)

# Tool 8 and 9 - Zoom in and out of Canvas
zoomin = Button(holder, text="Zoom In", height=1, width=int(WIDTH/4 - 3), command=zoomControl)
zoomin.grid(row=2, column=2)


zoomout= Button(holder, text="Zoom Out", height=1, width=int(WIDTH/4 - 3), command=zoomControl)
zoomout.grid(row=3, column=2)


#### Stroke Size ####

# Label for 10, 11, 12, 13
label8910 = Label(holder, text="STROKE SIZE", borderwidth=1, relief=SOLID, width=int(WIDTH/4))
label8910.grid(row=0, column=3)

# Tool 10 - Increament by 1
sizeiButton = Button(holder, text="Increase", height=1, width=int(WIDTH/4 - 3), command=strokeI)
sizeiButton.grid(row=1, column=3)

# Tool 11 - Decreament by 1
sizedButton = Button(holder, text="Decrease", height=1, width=int(WIDTH/4 - 3), command=strokeD)
sizedButton.grid(row=2, column=3)

'''
# Tool 12 - Default
defaultButton = Button(holder, text="Default", height=1, width=int(WIDTH/4 - 3), command=strokeDf)
defaultButton.grid(row=3, column=3)

'''

# Tool 13 - Speech-Draw
DimensionButton = Button(
    holder, text="speech-Draw", height=1, width=int(WIDTH/4 - 3), command=speak
)
DimensionButton.grid(row=3, column=3)


#### Canvas Frame ####

# Main Frame
frame2 = Frame(root, height=500, width=800)
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

root.bind("<t>", lambda event: speak())
root.bind("<l>", lambda event: lasso())
root.bind("<Left>", lambda event: moveLassoObject("Left"))
root.bind("<Right>", lambda event: moveLassoObject("Right"))
root.bind("<Up>", lambda event: moveLassoObject("Up"))
root.bind("<Down>", lambda event: moveLassoObject("Down"))
root.bind("<i>", lambda event: zoom(event, 2))
root.bind("<o>", lambda event: zoom(event, 0.5))
root.bind("<n>", lambda event: cycleTool())

########### Main Loop ###########
canvas.pack()
root.mainloop()



