import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import requests

class FoodDetectionApp:
    def __init__(self, window, window_title, video_width=640, video_height=480):
        self.window = window
        self.window.title(window_title)

        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, video_width)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, video_height)

        self.canvas = tk.Canvas(window, width=video_width, height=video_height)
        self.canvas.pack()

        self.delay = 10

        # Load YOLO
        self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

        # Define custom class mapping with names of fruits, vegetables, and food items
        self.classes = ["apple", "banana", "orange", "strawberry", "grape",
                        "watermelon", "pineapple", "kiwi", "pear", "peach",
                        "plum", "lemon", "lime", "cherry", "blueberry",
                        "raspberry", "avocado", "mango", "pomegranate",
                        "papaya", "apricot", "coconut", "fig", "dragonfruit",
                        "guava", "melon", "cantaloupe", "honeydew",
                        "cucumber", "carrot", "broccoli", "cauliflower",
                        "spinach", "lettuce", "tomato", "potato", "onion",
                        "garlic", "pepper", "corn", "bean", "pea",
                        "eggplant", "zucchini", "squash", "pumpkin",
                        "sweet potato", "yam", "mushroom", "olive",
                        "asparagus", "artichoke", "celery", "kale",
                        "turnip", "beet", "radish", "rutabaga",
                        "sauerkraut", "leek", "scallion", "chive",
                        "shallot", "fennel", "okra", "bean sprout",
                        "watercress", "arugula", "rhubarb", "endive",
                        "cabbage", "cactus", "bamboo", "chestnut",
                        "quince", "clementine", "date", "elderberry",
                        "gooseberry", "jackfruit", "lychee", "mulberry",
                        "persimmon", "quince", "tangerine", "salad",
                        "sandwich", "pizza", "burger", "hot dog",
                        "fries", "taco", "burrito", "sushi",
                        "pasta", "rice", "soup", "stew",
                        "curry", "noodle", "bread", "pancake",
                        "waffle", "muffin", "cake", "cookie",
                        "brownie", "pie", "pudding", "ice cream"]

        self.output_layers = ['yolo_82', 'yolo_94', 'yolo_106']  # Output layers of YOLOv3

        self.update()  # Call update method to start video stream

        self.window.mainloop()

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            # Detecting objects
            height, width, channels = frame.shape
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)

            # Showing information on the screen
            class_ids = []
            confidences = []
            boxes = []
            food_labels = []  # List to store food labels for calories lookup
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
                        food_labels.append(self.classes[class_id])

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            for i in range(len(boxes)):
                if i in indexes:
                    label = str(self.classes[class_ids[i]])

                    # Get nutritional information
                    calories = self.get_nutritional_info(food_labels[i])
                    if calories is not None:
                        label += f" ({calories} cal)"

                    x, y, w, h = boxes[i]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(self.delay, self.update)

    # Function to fetch nutritional information from USDA FoodData Central API
    def get_nutritional_info(self, food_name):
        # API endpoint and parameters
        api_url = "https://api.nal.usda.gov/fdc/v1/foods/search"
        api_key = "Raxd89emq6gRAkjrasg7xlFjfcADRjxDfOnu1RM4"  # Replace with your actual API key
        params = {
            "query": food_name,
            "api_key": api_key
        }

        # Make API request
        response = requests.get(api_url, params=params)

        # Parse API response and extract nutritional information
        if response.status_code == 200:
            data = response.json()
            if "foods" in data and data["foods"]:
                # Extract nutritional information from the first result
                food = data["foods"][0]
                calories = food["foodNutrients"][0]["value"] if "foodNutrients" in food else None
                return calories
        return None

def main():
    FoodDetectionApp(tk.Tk(), "Food Detection", video_width=640, video_height=480)

if __name__ == '__main__':
    main()
