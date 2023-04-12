import os
import cv2
import pydicom as dicom
from pathlib import Path
from inference import Segmentator
from tkinter import *
from tkinter import filedialog


class Application:
    def __init__(self, path_model):
        self.path_model = path_model
        self.listdir_single = None
        self.pathdir_save = None
        self.list_processed = []
        self.nn = Segmentator(path_model=self.path_model)

    def main(self):
        def clicked_select() -> tuple:
            self.listdir_single = filedialog.askopenfilenames()
            if self.listdir_single:
                var.set("Progress 0%")
                button_save_images.config(fg="red")
                button_select_images.config(fg="green")
                button_select_images_from_dir.config(fg="red")
                button_save_images.configure(state="normal")
                images_list.delete('1.0', END)
                images_list.insert(END,self.listdir_single)
                root.update_idletasks()  
        
        def clicked_select_dir_images() -> list:
            temp = filedialog.askdirectory()
            self.listdir_single = list(Path(temp).iterdir())
            if self.listdir_single:
                var.set("Progress 0%")
                button_select_images.config(fg="red")
                button_select_images_from_dir.config(fg="green")
                button_save_images.configure(state="normal")
                images_list.delete('1.0', END)
                images_list.insert(END,self.listdir_single)
                root.update_idletasks()

        def clicked_select_dir_save():
            temp_dir = filedialog.askdirectory()
            self.pathdir_save = Path(temp_dir).absolute().as_posix()
            if self.pathdir_save:
                button_save_images.config(fg="green")
                button_processing.configure(state="normal")
                root.update_idletasks()
            
        def processing() -> list:
            for i, path_images in enumerate(self.listdir_single):
                path_images = Path(path_images)
                if path_images.name[-4] != ".":
                    ds = dicom.dcmread(path_images)
                    image = ds.pixel_array
                else:
                    image = cv2.imread(path_images.absolute().as_posix())
                preds = self.nn.predict(image)
                new_filename = path_images.name[:-4]+'_pred.jpg'
                cv2.imwrite(os.path.join(self.pathdir_save, new_filename), preds)
                var.set("Progress {:6.2f}%".format((i + 1) / len(self.listdir_single) * 100))
                doDisable()
                button_select_images.config(fg='red')
                button_select_images_from_dir.config(fg='red')
                root.update_idletasks()
        def doDisable():
            button_processing.configure(state=DISABLED)
            button_save_images.configure(state=DISABLED)
                
        #define application
        root = Tk()
        root.title("Cancer Detector")
        root.geometry('900x200')
        
        images_list = Text(
            root,
            font=("Arial", 14),
            width=60,
            height=2
        )
        images_list.grid(column=2, row=0)
        images_list.config(font=("consolas", 12), undo=True, wrap=NONE)
        scrollb = Scrollbar(root,orient='horizontal', command=images_list.xview)
        scrollb.grid(row=1, column=2, sticky='nsew')
        images_list['xscrollcommand'] = scrollb.set
        #create label
        lbl_first_version = Label(
            root, 
            text="Select images:", 
            font=("Arial", 14)
        )
        lbl_first_version.grid(column=0, row=0, sticky='nesw')
        
        lbl_second_version = Label(
            root, 
            text="Select directory:", 
            font=("Arial", 14)
        )
        lbl_second_version.grid(column=0, row=1, sticky='nesw')       
        
        #define button for select folder
        button_select_images = Button(
            root,
            text="Select Images",
            fg='red',
            font=("Arial", 14),
            command=clicked_select
        )
        button_select_images.grid(column=1, row=0, sticky='nesw')

        button_select_images_from_dir = Button(
            root,
            text="Select Folder",
            fg='red',
            font=("Arial", 14),
            command=clicked_select_dir_images
        )
        button_select_images_from_dir.grid(column=1, row=1, sticky='nesw')
        
        button_save_images = Button(
            root,
            text="Directory Save",
            fg="red",
            font=("Arial", 14),
            command=clicked_select_dir_save
        )
        button_save_images.grid(column=0, row=3, sticky='nesw')  
        
        var = StringVar()
        var.set("Progress 0%")
        lbl_status_first = Label(
            root,
            textvariable=var,
            font=("Arial", 14)
        )
        lbl_status_first.grid(column=0, row=5)

        button_processing = Button(
            root,
            text="Processing",
            fg="Black",
            font=("Arial", 14),
            command=processing
        )
        button_processing.grid(column=0, row=4, sticky='nesw')
        doDisable()
           
        root.mainloop()       

if __name__=="__main__":
    print("Start work")
    print("Based")
    app = Application(Path('./model.pth').absolute().as_posix())
    app.main()