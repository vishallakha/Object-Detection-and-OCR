## Description
In this project, I was reading the CCTV feed(Images) of traffic Junctions, and performing the object detection task to identify the license plates for cars, motorcycles, auto rickshaws, trucks etc. I have used Tiny Yolo model as its very fast, lightweight and accurate enough to handle the load efficiently.

Once the license plate region has been identified, the cropped image is passed to the custom trained OCR model to extract out its license number. 

This data is stored in Mongo DB, along with image ID, creation datetime. this information is passed to the city traffic police department, who handles the endpoint of issuing the violation tickets to the home address of the violators.

## Technology Stack
Python, C++, CNN, LSTM, CTC, YOLO, Object Detection, OCR, CV2, Mongo DB