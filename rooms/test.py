import pickle

with open("/home/xin3/Desktop/stretch_ai_xin/rooms/Room1.pkl", "rb") as f:
    data = pickle.load(f)  # Loads a dictionary

# Access frames via dictionary key:
print(data.keys())
if "frames" in data:
    for frame in data["frames"]:
        print(frame)
else:
    print("Key 'frames' not found in the .pkl file.")