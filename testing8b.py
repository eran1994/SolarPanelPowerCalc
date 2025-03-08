import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox, Label, NW
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2
from threading import Thread
import numpy as np
from PIL import Image, ImageTk #You will need to install Pillow: pip install Pillow

resoultion = 33.3
def generate_hillshade(image, azimuth=315, altitude=45):
    """Generate a hillshade image using a simple algorithm with light direction."""
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute gradients
    sobelx = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate slope and aspect
    slope = np.arctan(np.sqrt(sobely ** 2 + sobelx ** 2))  # Use sqrt of the sum of squares for slope
    aspect = np.arctan2(sobely, -sobelx)  # Aspect correctly defined with y and x gradients

    # Convert altitude and azimuth to radians
    zenith_rad = np.radians(90 - altitude)
    azimuth_rad = np.radians(azimuth)

    # Calculate hillshade
    hillshade = (((np.cos(zenith_rad) * np.cos(slope)) + \
                (np.sin(zenith_rad) * np.sin(slope) * np.cos(azimuth_rad - aspect))) + 1) *255/2

    # Save and return the hillshade image
    cv2.imwrite("hillshade.jpg", hillshade)
    return hillshade


def filter_predictions_by_size(predictions, hillshade_path, use_hillshade_equation=False, prediction_image=None):
    filtered_predictions_small = []
    filtered_predictions_medium = []
    filtered_predictions_big = []
    total_area_house = 0
    total_area_residual = 0
    total_area_big = 0
    sunpower = 1367
    solarefficiency = 0.27

    # Load or generate hillshade
    if use_hillshade_equation and prediction_image is not None:
        prediction_img = cv2.imread(prediction_image)
        hillshade = generate_hillshade(prediction_img)
    else:
        hillshade = cv2.imread(hillshade_path, cv2.IMREAD_GRAYSCALE)

    for pred in predictions:
        bbox_width = pred.bbox.maxx - pred.bbox.minx
        bbox_height = pred.bbox.maxy - pred.bbox.miny
        area = bbox_width * bbox_height / (resoultion * resoultion)
        if area < 1000 and pred.score.value > 0.4: #0.008
            filtered_predictions_small.append(pred)
            for i in range(int(pred.bbox.minx), int(pred.bbox.maxx)):
                for j in range(int(pred.bbox.miny), int(pred.bbox.maxy)):
                    total_area_house += (hillshade[j, i] / 255) * sunpower * solarefficiency / (resoultion * resoultion)
                    hillshade[j, i] = 0
        elif 1000 < area < 10000 and pred.score.value > 0.4: #0.08
            filtered_predictions_medium.append(pred)
            for i in range(int(pred.bbox.minx), int(pred.bbox.maxx)):
                for j in range(int(pred.bbox.miny), int(pred.bbox.maxy)):
                    total_area_residual += (hillshade[j, i] / 255) * sunpower * solarefficiency / (resoultion * resoultion)
                    hillshade[j, i] = 0
        elif area > 10000 and pred.score.value > 0.4: #0.9
            filtered_predictions_big.append(pred)
            for i in range(int(pred.bbox.minx), int(pred.bbox.maxx)):
                for j in range(int(pred.bbox.miny), int(pred.bbox.maxy)):
                    total_area_big += (hillshade[j, i] / 255) * sunpower * solarefficiency / (resoultion * resoultion)
                    hillshade[j, i] = 0

    return filtered_predictions_small, filtered_predictions_medium, filtered_predictions_big, total_area_house, total_area_residual, total_area_big


def run_gui():
    def browse_prediction_image():
        prediction_image.set(filedialog.askopenfilename(filetypes=[("Image Files", "*.jpeg;*.jpg;*.png;*.tif")]))

    def browse_hillshade_image():
        hillshade_image.set(filedialog.askopenfilename(filetypes=[("Image Files", "*.jpeg;*.jpg;*.png;*.tif")]))

    def browse_output_image():
        output_image.set(filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG Image", "*.jpg")]))

    def start_analysis():
        progress_bar.start(100)  # Start the progress bar
        run_button.config(state=DISABLED)  # Disable the button during the process
        Thread(target=run_analysis).start()  # Run the analysis in a separate thread

    def run_analysis():
        try:
            prediction_file = prediction_image.get()
            hillshade_file = hillshade_image.get()
            output_file = output_image.get()

            if not prediction_file or not output_file:
                messagebox.showerror("Error", "Please select all required files.")
                progress_bar.stop()
                run_button.config(state=NORMAL)  # Re-enable the button
                return

            # Hardcoded model and settings
            model_file = "runs/detect/train18/weights/best.pt"
            confidence = 0.4
            detection_model = AutoDetectionModel.from_pretrained(
                model_type='yolov11',
                model_path=model_file,
                confidence_threshold=confidence,
                device='cuda'
            )

            # Perform predictions
            result = get_sliced_prediction(
                prediction_file,
                detection_model=detection_model,
                slice_height=256,
                slice_width=256,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,

            )

            # Determine if hillshade is to be generated
            use_hillshade_equation = not bool(hillshade_file)

            # Filter predictions using the hillshade image or equation
            filtered_predictions_small, filtered_predictions_medium, filtered_predictions_big, total_power_house, total_power_residual, total_area_big = filter_predictions_by_size(
                result.object_prediction_list,
                hillshade_file,
                use_hillshade_equation=use_hillshade_equation,
                prediction_image=prediction_file
            )

            # Save filtered predictions on the original image
            image = cv2.imread(prediction_file)
            for pred in filtered_predictions_small:
                bbox = pred.bbox
                cv2.rectangle(image, (int(bbox.minx), int(bbox.miny)), (int(bbox.maxx), int(bbox.maxy)), (255, 0, 0), 2)
            for pred in filtered_predictions_medium:
                bbox = pred.bbox
                cv2.rectangle(image, (int(bbox.minx), int(bbox.miny)), (int(bbox.maxx), int(bbox.maxy)), (0, 255, 0), 2)
            for pred in filtered_predictions_big:
                bbox = pred.bbox
                cv2.rectangle(image, (int(bbox.minx), int(bbox.miny)), (int(bbox.maxx), int(bbox.maxy)), (0, 0, 255), 2)
            cv2.imwrite(output_file, image)

            # Display results in the GUI
            results_text.set(
                f"Filtered {len(filtered_predictions_small)} house predictions.\n"
                f"Filtered {len(filtered_predictions_medium)} school/city building predictions.\n"
                f"Filtered {len(filtered_predictions_big)} factory predictions.\n"
                f"Total power of house filtered predictions: {total_power_house:.2f}[W]\n"
                f"Total power of school/city building filtered predictions: {total_power_residual:.2f}[W]\n"
                f"Total power of factory filtered predictions: {total_area_big:.2f}[W]\n"
                f"Total power for house and school/city building: {(total_power_house + total_power_residual):.2f}[W]\n"
                f"Total power: {(total_area_big + total_power_house + total_power_residual):.2f}[W]"
            )
            # Load and display the input image
            input_img = Image.open(prediction_file)
            input_img.thumbnail((300, 300))  # Resize for display
            input_photo = ImageTk.PhotoImage(input_img)
            input_image_label.config(image=input_photo)
            input_image_label.image = input_photo  # Keep a reference!

            # Load and display the output image
            output_img = Image.open(output_file)
            output_img.thumbnail((300, 300))  # Resize for display
            output_photo = ImageTk.PhotoImage(output_img)
            output_image_label.config(image=output_photo)
            output_image_label.image = output_photo  # Keep a reference!
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        finally:
            progress_bar.stop()  # Stop the progress bar
            run_button.config(state=NORMAL)  # Re-enable the button

    # Create the main window
    app = ttk.Window(themename="darkly")
    app.title("Solar Panel Analyzer")
    app.geometry("800x800")

    # Variables
    prediction_image = ttk.StringVar()
    hillshade_image = ttk.StringVar()
    output_image = ttk.StringVar()
    results_text = ttk.StringVar()

    # Layout
    ttk.Label(app, text="Prediction Image:", font=("Helvetica", 12)).grid(row=0, column=0, padx=10, pady=10, sticky="e")
    ttk.Entry(app, textvariable=prediction_image, width=40).grid(row=0, column=1, padx=10, pady=10)
    ttk.Button(app, text="Browse", bootstyle="primary", command=browse_prediction_image).grid(row=0, column=2, padx=10, pady=10)

    ttk.Label(app, text="Hillshade Image (Optional):", font=("Helvetica", 12)).grid(row=1, column=0, padx=10, pady=10, sticky="e")
    ttk.Entry(app, textvariable=hillshade_image, width=40).grid(row=1, column=1, padx=10, pady=10)
    ttk.Button(app, text="Browse", bootstyle="primary", command=browse_hillshade_image).grid(row=1, column=2, padx=10, pady=10)

    ttk.Label(app, text="Output Image:", font=("Helvetica", 12)).grid(row=2, column=0, padx=10, pady=10, sticky="e")
    ttk.Entry(app, textvariable=output_image, width=40).grid(row=2, column=1, padx=10, pady=10)
    ttk.Button(app, text="Browse", bootstyle="primary", command=browse_output_image).grid(row=2, column=2, padx=10, pady=10)

    run_button = ttk.Button(app, text="Run Analysis", bootstyle="success", command=start_analysis)
    run_button.grid(row=3, column=1, pady=20)

    progress_bar = ttk.Progressbar(app, mode="indeterminate", bootstyle="info")
    progress_bar.grid(row=4, column=1, pady=10)
    ttk.Label(app, textvariable=results_text, font=("Helvetica", 12), anchor="w", justify="left").place(x=100,y=600)
    input_image_label = Label(app, text="")
    output_image_label = Label(app, text="")
    input_image_label.place(x=50, y=250)
    output_image_label.place(x=400, y=250)
    # Start the application
    app.mainloop()


# Run the GUI
run_gui()