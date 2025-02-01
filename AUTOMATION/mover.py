import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

# Function to move files based on selected format
def move_files():
    source_folder = source_entry.get()
    target_folder = target_entry.get()
    file_format = format_entry.get()

    if not source_folder or not target_folder or not file_format:
        messagebox.showwarning("Input Error", "Please fill in all fields.")
        return

    if not os.path.exists(source_folder):
        messagebox.showerror("Error", "Source folder does not exist.")
        return

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    moved_files = 0
    for file_name in os.listdir(source_folder):
        if file_name.endswith(file_format):
            shutil.move(os.path.join(source_folder, file_name), os.path.join(target_folder, file_name))
            moved_files += 1

    messagebox.showinfo("Success", f"Moved {moved_files} files.")

# Function to browse source folder
def browse_source():
    folder_selected = filedialog.askdirectory()
    source_entry.delete(0, tk.END)
    source_entry.insert(0, folder_selected)

# Function to browse target folder
def browse_target():
    folder_selected = filedialog.askdirectory()
    target_entry.delete(0, tk.END)
    target_entry.insert(0, folder_selected)

# Setting up the GUI
root = tk.Tk()
root.title("File Mover Tool")
root.geometry("500x350")
root.configure(bg="#e6f7ff")

# Styling
style = ttk.Style()
style.theme_use("clam")  # Use a modern theme
style.configure("TButton", padding=6, relief="flat", background="#007acc", foreground="white", font=("Arial", 10, "bold"))
style.map("TButton", background=[("active", "#005f99")])
style.configure("TLabel", background="#e6f7ff", font=("Arial", 11, "bold"), foreground="#333")
style.configure("TEntry", padding=5, relief="solid")

# Frame for better layout
frame = ttk.Frame(root, padding=20, style="TFrame")
frame.pack(expand=True, fill="both")

# Header
header_label = tk.Label(frame, text="📁 File Mover Tool", font=("Arial", 16, "bold"), bg="#007acc", fg="white", pady=10)
header_label.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))

# Source folder
source_label = ttk.Label(frame, text="Source Folder:")
source_label.grid(row=1, column=0, sticky="w", pady=5)
source_entry = ttk.Entry(frame, width=40)
source_entry.grid(row=2, column=0, padx=5, pady=5)
source_button = ttk.Button(frame, text="Browse", command=browse_source)
source_button.grid(row=2, column=1, padx=5)

# Target folder
target_label = ttk.Label(frame, text="Target Folder:")
target_label.grid(row=3, column=0, sticky="w", pady=5)
target_entry = ttk.Entry(frame, width=40)
target_entry.grid(row=4, column=0, padx=5, pady=5)
target_button = ttk.Button(frame, text="Browse", command=browse_target)
target_button.grid(row=4, column=1, padx=5)

# File format
format_label = ttk.Label(frame, text="File Format (e.g., .pdf, .csv):")
format_label.grid(row=5, column=0, sticky="w", pady=5)
format_entry = ttk.Entry(frame, width=20)
format_entry.grid(row=6, column=0, padx=5, pady=5)

# Move button
move_button = ttk.Button(frame, text="🚀 Move Files", command=move_files)
move_button.grid(row=7, column=0, columnspan=2, pady=20, sticky="ew")

root.mainloop()
