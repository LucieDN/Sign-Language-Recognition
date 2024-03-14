from difflib import get_close_matches
import tkinter
from tkinter import Frame


mainapp = tkinter.Tk()
mainapp.title("Sign language dictionnary")
mainapp.minsize(1000,600)
mainapp.resizable(width=False, height=False)

# Create the structure
row_frame = tkinter.Frame(mainapp)
row_frame.columnconfigure(0, weight=1)
row_frame.columnconfigure(1, weight=1)
row_frame.grid(row=0, column=0, sticky="nsew")

top_frame = Frame(row_frame, width=1000, height=100, bg='blue')
top_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
top_frame.grid_propagate(0)

column_frame = tkinter.Frame(row_frame)
column_frame.columnconfigure(0, weight=1)
column_frame.columnconfigure(1, weight=1)
column_frame.grid(row=1, column=0)

# Create left and right frames
left_frame = Frame(column_frame, width=300, height=500, bg='red')
left_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
left_frame.grid_propagate(0)
left_frame.pack_propagate(0)

right_frame = Frame(column_frame, width=700, height=500, bg='green')
right_frame.grid(row=1, column=1, padx=10, pady=5, sticky="nsew")
right_frame.grid_propagate(0)
right_frame.pack_propagate(0)

# Create the searching barre
searching_frame = Frame(left_frame, width=300, height=50, bg='grey')
searching_frame.rowconfigure(1, weight=1)
searching_frame.rowconfigure(1, weight=1)
searching_frame.columnconfigure(1, weight=1)
searching_frame.columnconfigure(1, weight=1)
searching_frame.grid()
searching_frame.grid_propagate(0)
searching_frame.pack_propagate(0)

# renvoie la valeur de la liste la plus proche
def updateEntry(entry, sv_closer, list):
    value = entry.get()
    print(value)
    matches = get_close_matches(value, list)
    if matches:
        sv_closer.set(matches[0])
    print(sv_closer.get())
    return sv_closer

sv = tkinter.StringVar()
closer= tkinter.StringVar()
list  =['ape', 'apple', 'peach', 'puppy']

entry = tkinter.Entry(searching_frame, textvariable=sv, validate="focusout")#, validate="focusout"
entry.grid(row = 0,column = 0,padx=5,pady=5, sticky="nsew")
button = tkinter.Button(searching_frame, text = "Rechercher", command = lambda: updateEntry(entry, closer, list))
button.grid(row = 0,column = 1,padx=5,pady=5, sticky="nsew")



word = tkinter.Label(left_frame, background="pink", textvariable=sv)
word.grid(row=1, column=0,padx=5,pady=5, sticky="wn")

mainapp.mainloop()

